"""extractor.py

LLM‑powered text / PDF parser for **value‑add multifamily** OMs & deal
emails.  Produces a clean JSON dict that can be fed into the underwriting
model (see build_model.py).

---------------------------------------------------------------------
Returned JSON schema
====================
{
  "Unit Mix":            { "<bed>/<bath>": <int units>, ... } | null,
  "Current Rents":       { "<bed>/<bath>": <float avg_monthly_rent>, ... } | null,
  "Unit Sizes SqFt":     { "<bed>/<bath>": <int avg_sqft>, ... } | null,
  "Asking Price":        float | null,
  "Year Built":          int | null,

  "Property Taxes (Annual)":    float | null,
  "Utility Expenses (Annual)":  float | null,
  "% Utilities Recovered":      float | null,
  "Other Income":               { "<src>": <float annual>, ... } | null
}

* Currency values are **annual USD** (omit $/commas).
* If you cannot confidently find a value, set that field to null.
* Return **only** the raw JSON – no markdown or commentary.

---------------------------------------------------------------------
Design notes
-----------
* Uses **pdfplumber** for PDF → text, falls back to raw plaintext.
* One single OpenAI chat call does the heavy extraction.
* Temperature 0, model = gpt‑4o‑mini (cheap + capable).

---------------------------------------------------------------------
Example
-------
>>> from extractor import parse_deal
>>> data = parse_deal(pdf_path="123_Main_OM.pdf")
>>> data["Unit Mix"]
{'1/1': 24, '2/1': 12}
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, Union

import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY env var missing.")

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a meticulous multifamily acquisitions analyst.
    Extract ONLY the fields listed in the schema below, following these strict rules:
    1. Use ONLY ACTUAL/CURRENT values - NEVER use proforma/projected/forecasted values
    2. For rents, use ONLY in-place/current rents - NOT market/projected/asking rents
    3. For expenses and income, use data in this priority order:
       a. T12 (trailing 12 months) if available
       b. T6 (trailing 6 months) annualized if T12 not available
       c. T3 (trailing 3 months) annualized if T6 not available
       d. YTD annualized if no trailing data available
       e. Set to null if only monthly snapshot or no clear timeframe available
    4. When annualizing partial year data:
       - T6: multiply by 2
       - T3: multiply by 4
       - YTD: divide by number of months, then multiply by 12
    5. If a field is missing or only proforma values are available, output null
    6. Do not hallucinate or estimate values
    7. Return a single, valid JSON object - no markdown fences
    8. All currency values must be annual USD with no $ signs or commas
    9. Unit types must be standardized as: "Studio", "1/1", "2/1", "2/2", etc.
    10. Other Income sources must be standardized as: "Laundry", "Parking", "Pet Rent", "Storage", "Late Fees", "Other"
    11. For Unit Mix, Current Rents, and Unit Sizes - ONLY include unit types that exist in the property
    12. Calculate "% Utilities Recovered" as: (Annual Utility Revenue / Annual Utility Expenses) * 100
    13. Calculate "Utility Billback Per Unit Monthly" as: (Annual Utility Revenue / Total Units / 12)
    """
)

SCHEMA_EXAMPLE = textwrap.dedent(
    """\
    Schema:
    Example schema with standardized formatting and required fields:
    {
      "Property Name": "Meadowbrook Apartments",
      "Property Address": "123 Main Street, Anytown, CA 90210",
      "Unit Mix": {                 # ONLY include unit types that exist
        "1/1": 24, 
        "2/2": 16
      },
      "Current Rents": {           # ONLY in-place/current rents for existing unit types
        "1/1": 950.0,
        "2/2": 1250.0
      },
      "Unit Sizes SqFt": {         # Actual unit sizes for existing unit types only
        "1/1": 725,
        "2/2": 1000
      },
      "Asking Price": 5750000.0,   # Current asking price, no formatting
      "Year Built": 1985,          # Original build year, not renovation year
      "Property Taxes (Annual)": 45000.0,      # Current tax bill, not projected
      "Utility Expenses (Annual)": 38000.0,    # Use T12, or annualized T6/T3/YTD in that order
      "% Utilities Recovered": 80.0,           # (Annual Utility Revenue / Annual Utility Expenses) * 100
      "Utility Billback Per Unit Monthly": 25.0,  # (Annual Utility Revenue / Total Units / 12)
      "Other Income": {                        # Use T12, or annualized T6/T3/YTD in that order
        "Utility Revenue": 30400.0,            # Annual utility reimbursement revenue
        "Laundry": 9600.0,
        "Parking": 12000.0,
        "Pet Rent": 3600.0,
        "Storage": 2400.0,
        "Late Fees": 1800.0,
        "Other": 1200.0
      }
    }
    """
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    with pdfplumber.open(str(pdf_path)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def _call_llm(raw_text: str) -> Dict[str, Any]:
    """Hit the ChatCompletion endpoint and parse JSON output."""
    user_prompt = (
        f"{SCHEMA_EXAMPLE}\n\n"
        "Below is the offering memorandum or email body.\n"
        "---\n"
        f"{raw_text}\n"
        "---\n\n"
        "Extract the JSON as per the schema.  JSON only."
    )

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = resp.choices[0].message.content.strip()
    # strip accidental ``` fences
    content = re.sub(r"^```json|```$", "", content).strip()
    return json.loads(content)

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def parse_deal(
    *,
    pdf_path: Union[str, Path, None] = None,
    plain_text: str | None = None,
    save_json_to: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    """Parse a PDF OM or plaintext email and return underwriting JSON."""
    if not (pdf_path or plain_text):
        raise ValueError("Provide either pdf_path or plain_text.")
    if pdf_path and plain_text:
        raise ValueError("Provide only one input.")

    text = plain_text if plain_text else _extract_text_from_pdf(pdf_path)  # type: ignore[arg-type]

    data = _call_llm(text)

    if save_json_to:
        Path(save_json_to).write_text(json.dumps(data, indent=2))

    return data


# ------------------------------------------------------------------
# CLI (for ad‑hoc use)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Extract multifamily OM data → JSON")
    p.add_argument("source", help="Path to PDF OM or .txt email file")
    p.add_argument("--out", "-o", help="Save extracted JSON here for review")
    args = p.parse_args()

    src = Path(args.source)
    if src.suffix.lower() == ".pdf":
        payload = parse_deal(pdf_path=src, save_json_to=args.out)
    else:
        payload = parse_deal(plain_text=src.read_text(), save_json_to=args.out)

    print(json.dumps(payload, indent=2))
