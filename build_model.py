"""build_model.py – multifamily value‑add v0.2

Populate the **multi‑template.xlsx** underwriting model with data
extracted by `extractor.py`.

Changes in this version
-----------------------
* Maps **all red input cells** in the template:
    · Summary!A3  –  Property Name  (new)
    · Summary!A4  –  Address        (new)
    · Summary!B5  –  Asking Price
    · Summary!B7  –  Purchase Price
    · Summary!B8  –  Hold Period
    · Summary!B6  –  Year Built
    · Unit schedule starting row 4 columns H–K:
        H  # Units          (data["Unit Mix"][type])
        I  Type (bed/bath) (dict key itself)
        J  Avg Sq Ft        (data["Unit Sizes SqFt"][type])
        K  Current Rent     (data["Current Rents"][type])
* Graceful error handling – any missing / null field writes **0** (for
  numerics) or blanks (for text) so the workbook never raises formula
  errors.
* Convenience default: if `--template` is omitted the script looks for
  `multi-template.xlsx` in the same folder as this file.

CLI examples
~~~~~~~~~~~~
$ python build_model.py MyDeal.pdf
$ python build_model.py MyEmail.txt  --out underwriting.xlsx
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import openpyxl as _xl

import extractor as _ex

# ------------------------------------------------------------------
# Constants & mapping tables
# ------------------------------------------------------------------

#   extractor‑key  →  (sheet, cell)
CELL_MAP: Dict[str, tuple[str, str]] = {
    "Property Name": ("Summary", "A3"),
    "Property Address": ("Summary", "A4"),
    "Asking Price": ("Summary", "B5"),
    "Year Built": ("Summary", "B6"),
    "Property Taxes (Annual)": ("Summary", "E26"),  # Fixed cell reference
    "Utility Expenses (Annual)": ("Summary", "E28"),  # Adjusted for proper sequence
    "% Utilities Recovered": ("Summary", "B24"),  # Current percent billback
    "Utility Billback Per Unit Monthly": ("Summary", "K12"),  # Utility billback per unit
}

# Special cells that are computed from other values
COMPUTED_CELLS = {
    "timestamp": ("Summary", "L1"),
    "utility_per_unit_monthly": ("Summary", "B23"), # = E28 / sum(Unit Mix)
    "other_income_sum": ("Summary", "B30"),
    "purchase_price": ("Summary", "B9"),      # Purchase price (90% of asking)
    "hold_period": ("Summary", "B8"),         # Hold period in years, 5 by default
    "property_tax_pct": ("Summary", "B21"),   # Property tax as % of purchase price
}

# Unit‑schedule columns (Summary sheet, starting row 4)
_COL_UNITS = "H"
_COL_TYPE = "I"
_COL_SIZE = "J"
_COL_RENT = "K"
_START_ROW = 4

# Default values when extractor cannot find a field
_DEFAULT_NUMERIC = 0
_DEFAULT_TEXT = ""

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def _calculate_utility_per_unit_monthly(data: dict[str, Any]) -> float:
    """Calculate utility expenses per unit per month."""
    utility_expenses = data.get("Utility Expenses (Annual)")
    unit_mix = data.get("Unit Mix", {})
    
    if not utility_expenses or not unit_mix:
        return 0.0
        
    total_units = sum(unit_mix.values())
    if total_units == 0:
        return 0.0
        
    return utility_expenses / total_units / 12


def _process_other_income(data: dict[str, Any]) -> float:
    """Extract and calculate sum of other income, excluding utility revenue.
    
    Returns:
        float: other_income_sum (excluding utility revenue)
    """
    other_income = data.get("Other Income", {})
    if not other_income:
        return 0.0
    
    # Sum all other income EXCEPT utility revenue
    other_income_sum = sum(
        amount or 0.0
        for category, amount in other_income.items()
        if amount is not None and category != "Utility Revenue"
    )
    
    return other_income_sum


def _get_merged_cell_master(sheet: _xl.worksheet.worksheet.Worksheet, cell_ref: str) -> tuple[str, _xl.cell.Cell]:
    """Get the master (top-left) cell of a merged range if cell is merged.
    
    Args:
        sheet: Worksheet to check
        cell_ref: Cell reference (e.g. 'A1')
    
    Returns:
        tuple[str, Cell]: (master cell reference, master cell object)
    """
    cell = sheet[cell_ref]
    
    if isinstance(cell, _xl.cell.MergedCell):
        for merged_range in sheet.merged_cells.ranges:
            if cell_ref in merged_range:
                master_ref = merged_range.start_cell.coordinate
                return master_ref, sheet[master_ref]
    
    return cell_ref, cell


def _validate_cell_writable(sheet: _xl.worksheet.worksheet.Worksheet, cell_ref: str) -> tuple[str, _xl.cell.Cell]:
    """Validate that a cell can be written to and get its writable reference.
    
    Args:
        sheet: Worksheet containing the cell
        cell_ref: Cell reference to validate
    
    Returns:
        tuple[str, Cell]: (writable cell reference, cell object)
    
    Raises:
        ValueError: If cell cannot be written to
    """
    cell_ref, cell = _get_merged_cell_master(sheet, cell_ref)
    
    # Verify the cell is actually writable
    try:
        current_value = cell.value
        cell.value = current_value  # Test write
        return cell_ref, cell
    except AttributeError as e:
        raise ValueError(f"Cell {cell_ref} is not writable: {e}")


def _safe_get(sheet: _xl.worksheet.worksheet.Worksheet, cell_ref: str) -> Any:
    """Safely read a cell value, handling merged cells.
    
    Args:
        sheet: Worksheet to read from
        cell_ref: Cell reference (e.g. 'A1')
    
    Returns:
        Any: Cell value
    """
    try:
        cell_ref, cell = _get_merged_cell_master(sheet, cell_ref)
        _debug_log(f"Reading from cell {cell_ref}: type={type(cell)}, value={cell.value}")
        return cell.value
    except Exception as e:
        _debug_log(f"Error reading from {cell_ref}: {e}")
        return _DEFAULT_NUMERIC


def _verify_cell_write(sheet: _xl.worksheet.worksheet.Worksheet, cell_ref: str, expected_value: Any) -> None:
    """Verify that a cell write was successful by reading back the value."""
    actual_value = _safe_get(sheet, cell_ref)
    if actual_value != expected_value:
        _debug_log(f"Write verification failed for {cell_ref}: expected={expected_value}, got={actual_value}")
    else:
        _debug_log(f"Write verification successful for {cell_ref}: value={actual_value}")


def _safe_set(sheet: _xl.worksheet.worksheet.Worksheet, cell_ref: str, value: Any) -> None:
    """Write to cell with comprehensive error handling.
    
    Args:
        sheet: Worksheet to write to
        cell_ref: Cell reference (e.g. 'A1')
        value: Value to write
    """
    try:
        # Get writable cell reference
        cell_ref, cell = _validate_cell_writable(sheet, cell_ref)
        _debug_log(f"Writing to cell {cell_ref}: type={type(cell)}")
        
        # Store original cell properties
        original_data_type = cell.data_type
        original_formula = cell.value if str(cell.value).startswith('=') else None
        
        # If value is None, preserve the existing cell value
        if value is None:
            _debug_log(f"Skipping write to {cell_ref} - preserving existing value: {cell.value}")
            return
            
        # Write the value while preserving data type
        cell.value = value
        
        # If it was a formula cell, ensure we're setting a raw value
        if original_formula:
            _debug_log(f"Converting formula cell {cell_ref} to raw value")
            cell.data_type = original_data_type
            cell.value = value
        
        # Force a read-back verification before moving on
        actual_value = _safe_get(sheet, cell_ref)
        if actual_value != value:
            _debug_log(f"Immediate verification failed for {cell_ref}: expected={value}, got={actual_value}")
            # Try one more time with explicit number format
            if isinstance(value, (int, float)):
                cell.number_format = '0.00'
                cell.value = float(value)
        
        _verify_cell_write(sheet, cell_ref, value)
        
    except ValueError as e:
        _debug_log(f"Warning: Could not write to {cell_ref}: {e}")
    except Exception as e:
        _debug_log(f"Warning: Unexpected error writing to {cell_ref}: {e}")


def _write_unit_table(wb: _xl.Workbook, data: dict[str, Any]) -> None:
    """Populate the variable‑length unit mix table in Summary!H–K and add calculations for L,M,N."""
    sheet = wb["Summary"]

    mix: Dict[str, int] | None = data.get("Unit Mix")
    rents: Dict[str, float] | None = data.get("Current Rents")
    sizes: Dict[str, int] | None = data.get("Unit Sizes SqFt")

    if not mix:
        return  # nothing to write

    # Count how many unit types we have
    unit_type_count = len(mix)

    # Write unit table data
    for idx, (utype, count) in enumerate(mix.items()):
        r = _START_ROW + idx
        
        if count is not None:
            _safe_set(sheet, f"{_COL_UNITS}{r}", count)
        if utype:
            _safe_set(sheet, f"{_COL_TYPE}{r}", utype)
        if sizes and utype in sizes and sizes[utype]:
            _safe_set(sheet, f"{_COL_SIZE}{r}", sizes[utype])
        if rents and utype in rents and rents[utype]:
            _safe_set(sheet, f"{_COL_RENT}{r}", rents[utype])
        
        # Add formulas for L, M, N columns if we have multiple unit types
        if unit_type_count > 1:
            # L column: Rent per SF = Current Rent / Avg SF
            _safe_set(sheet, f"L{r}", f"=+K{r}/J{r}")
            
            # M column: Market Rent = Current Rent * (1 + Rent Growth)^Hold Period
            _safe_set(sheet, f"M{r}", f"=+K{r}*(1+$B$20)^$B$8")
            
            # N column: Market Rent per SF = Market Rent / Avg SF
            _safe_set(sheet, f"N{r}", f"=+M{r}/J{r}")

    _debug_log(f"Wrote unit table with {unit_type_count} types, added LMN formulas: {unit_type_count > 1}")


def _debug_log(msg: str) -> None:
    """Print debug information if DEBUG environment variable is set."""
    if os.environ.get("DEBUG"):
        print(f"DEBUG: {msg}")


def _validate_cell_mapping() -> None:
    """Validate that there are no conflicts in cell mappings."""
    all_cells = set()
    
    # Check CELL_MAP
    for key, (sheet, cell) in CELL_MAP.items():
        cell_id = (sheet, cell)
        if cell_id in all_cells:
            raise ValueError(f"Duplicate cell mapping found: {sheet}!{cell}")
        all_cells.add(cell_id)
    
    # Check computed cells
    for key, (sheet, cell) in COMPUTED_CELLS.items():
        cell_id = (sheet, cell)
        if cell_id in all_cells:
            raise ValueError(f"Computed cell {key} ({sheet}!{cell}) conflicts with mapped cells")
        all_cells.add(cell_id)
    
    _debug_log(f"Validated {len(all_cells)} unique cell mappings")


def _calculate_property_tax_percent(annual_tax: float, purchase_price: float) -> float:
    """Calculate property tax as a percentage of purchase price.
    Returns the decimal value (not percentage) for Excel percentage-formatted cells."""
    if not purchase_price or purchase_price == 0:
        return 0.0
    return (annual_tax / purchase_price)  # Return decimal, not percentage


def _calculate_utility_billback_percent(data: dict[str, Any]) -> float:
    """Calculate utility billback percentage.
    Returns the decimal value (not percentage) for Excel percentage-formatted cells."""
    other_income = data.get("Other Income", {})
    utility_expenses = data.get("Utility Expenses (Annual)", 0.0)
    
    if not other_income or not utility_expenses:
        return 0.0
        
    utility_revenue = other_income.get("Utility Revenue", 0.0) or 0.0
    
    if utility_expenses == 0:
        return 0.0
        
    return (utility_revenue / utility_expenses)  # Return decimal, not percentage


def _populate_workbook(template: Path, out: Path, data: dict[str, Any]) -> Path:
    # Validate cell mappings first
    _validate_cell_mapping()
    
    wb = _xl.load_workbook(template)
    sheet = wb["Summary"]  # We'll only work with Summary sheet

    # Validate template structure
    try:
        # Test all mapped cells
        for key, (_, cell) in CELL_MAP.items():
            _validate_cell_writable(sheet, cell)
        
        # Test all computed cells
        for key, (_, cell) in COMPUTED_CELLS.items():
            _validate_cell_writable(sheet, cell)
            
    except ValueError as e:
        raise ValueError(f"Template validation failed: {e}")

    _debug_log("Template structure validated successfully")

    purchase_price = None
    annual_property_tax = None

    # Write only fields that exist in the data
    for key, (sheet_name, cell) in CELL_MAP.items():
        if key in data and data[key] is not None:
            value = data[key]
            
            # Special handling for purchase price
            if key == "Asking Price":
                asking_price = value
                # Set purchase price to 90% of asking price
                purchase_price = asking_price * 0.9
                _debug_log(f"Setting purchase price to 90% of asking price: {purchase_price}")
                _safe_set(sheet, COMPUTED_CELLS["purchase_price"][1], purchase_price)
            
            # Store property tax value for later calculation
            elif key == "Property Taxes (Annual)":
                annual_property_tax = value
            
            # Special handling for utility billback percentage
            elif key == "% Utilities Recovered":
                billback_pct = _calculate_utility_billback_percent(data)
                _debug_log(f"Setting utility billback percentage (decimal): {billback_pct}")
                _safe_set(sheet, cell, billback_pct)
                continue
            
            _debug_log(f"Writing {key}={value} to {sheet_name}!{cell}")
            _safe_set(sheet, cell, value)
    
    # Calculate and write property tax percentage if we have both required values
    if purchase_price and annual_property_tax:
        tax_percent = _calculate_property_tax_percent(annual_property_tax, purchase_price)
        _debug_log(f"Calculated property tax percentage (decimal): {tax_percent}")
        _safe_set(sheet, COMPUTED_CELLS["property_tax_pct"][1], tax_percent)

    # Set default hold period (5 years)
    hold_period_cell = COMPUTED_CELLS["hold_period"][1]
    _debug_log("Setting default hold period to 5 years")
    _safe_set(sheet, hold_period_cell, 5)
    
    # Calculate and write utility expense per unit per month if we have the required data
    if "Utility Expenses (Annual)" in data and "Unit Mix" in data:
        utility_per_unit = _calculate_utility_per_unit_monthly(data)
        _debug_log(f"Writing utility per unit monthly={utility_per_unit} to Summary!B23")
        _safe_set(sheet, COMPUTED_CELLS["utility_per_unit_monthly"][1], utility_per_unit)

    # Process and write other income data only if we have it
    if "Other Income" in data:
        other_income_sum = _process_other_income(data)
        if other_income_sum:
            _debug_log(f"Writing other income sum={other_income_sum} to Summary!B30")
            _safe_set(sheet, COMPUTED_CELLS["other_income_sum"][1], other_income_sum)

    # Write unit table only if we have unit mix data
    if "Unit Mix" in data:
        _write_unit_table(wb, data)

    # Write timestamp
    timestamp = _dt.datetime.now().strftime("%Y‑%m‑%d %H:%M")
    _debug_log(f"Writing timestamp={timestamp} to {COMPUTED_CELLS['timestamp'][0]}!{COMPUTED_CELLS['timestamp'][1]}")
    _safe_set(sheet, COMPUTED_CELLS["timestamp"][1], timestamp)

    # Save workbook
    wb.save(out)
    return out

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Fill multi‑template.xlsx with extracted data")
    parser.add_argument("source", help="OM PDF path **or** .txt email file")
    parser.add_argument("--template", help="Excel template (.xlsx). Defaults to multi-template.xlsx next to this script")
    parser.add_argument("--out", help="Output path (.xlsx). Default: <source>_underwriting.xlsx")
    parser.add_argument("--debug-json", action="store_true", help="Print extracted JSON data before processing")
    args = parser.parse_args(argv)

    src = Path(args.source)
    template = Path(args.template) if args.template else Path(__file__).with_name("multi-template-v2.xlsx")
    if not template.exists():
        sys.exit(f"✖ Template not found: {template}")

    out = Path(args.out) if args.out else src.with_name(src.stem + "_underwriting.xlsx")

    # run extractor ------------------------------------------------
    if src.suffix.lower() == ".pdf":
        data = _ex.parse_deal(pdf_path=src)
    else:
        data = _ex.parse_deal(plain_text=src.read_text())

    # Print extracted data in a readable format
    import json
    print("\n=== Extracted Data ===")
    print(json.dumps(data, indent=2, default=str))
    print("===================\n")

    # build workbook ----------------------------------------------
    built_path = _populate_workbook(template, out, data)
    print(f"✔ Underwriting model written: {built_path}")


if __name__ == "__main__":
    main()
