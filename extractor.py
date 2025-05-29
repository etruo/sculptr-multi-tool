"""extractor.py

LLM‑powered text / PDF parser for **value‑add multifamily** OMs & deal
emails.  Produces a clean JSON dict that can be fed into the underwriting
model (see build_model.py).

Now with OCR support for image-based PDFs.

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
import tempfile
from pathlib import Path
from typing import Any, Dict, Union, List

import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance
from dotenv import load_dotenv
from openai import OpenAI
import cv2
import numpy as np

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

def _preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """Apply advanced image preprocessing for better OCR results."""
    # Convert PIL Image to cv2 format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Apply adaptive thresholding
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh)
    
    # Convert back to PIL
    enhanced_img = Image.fromarray(denoised)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(enhanced_img)
    enhanced_img = enhancer.enhance(2.0)
    
    # Increase resolution
    width, height = enhanced_img.size
    scale_factor = 2
    enhanced_img = enhanced_img.resize(
        (width * scale_factor, height * scale_factor),
        Image.Resampling.LANCZOS
    )
    
    return enhanced_img

def _extract_text_from_image(image: Image) -> str:
    """Extract text from an image using OCR with enhanced preprocessing."""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply preprocessing
        processed_image = _preprocess_image_for_ocr(image)
        
        # OCR Configuration
        custom_config = r'''--oem 3 
            --psm 6 
            -c preserve_interword_spaces=1
            -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,/$%- "
            -c tessedit_pageseg_mode=6
            -c textord_heavy_nr=1
            -c textord_min_linesize=2.5
        '''
        
        # Extract text with detailed configuration
        text = pytesseract.image_to_string(
            processed_image,
            config=custom_config,
            lang='eng'
        )
        
        if os.environ.get("DEBUG"):
            print(f"OCR extracted text length: {len(text)}")
            print("Sample of extracted text:", text[:200])
            # Save debug images
            debug_dir = "debug_ocr"
            os.makedirs(debug_dir, exist_ok=True)
            processed_image.save(f"{debug_dir}/processed_page.png")
            
        return text
    except Exception as e:
        print(f"OCR error: {e}")
        return ""

def _extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    """Extract text from PDF using both pdfplumber and OCR if needed."""
    text_content = []
    debug = os.environ.get("DEBUG", False)
    
    # First try pdfplumber for native text extraction
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            if debug:
                print(f"Page {page_num} native text extraction length: {len(text)}")
            text_content.append(text)
    
    # If we got very little text, try OCR
    if not any(text_content) or sum(len(t) for t in text_content) < 100:
        if debug:
            print("Limited text extracted, attempting OCR...")
        
        try:
            # Convert PDF to images with higher DPI for better quality
            images = convert_from_path(
                pdf_path,
                dpi=400,  # Increased DPI
                fmt='png',  # Using PNG for better quality
                grayscale=True,  # Convert to grayscale
                thread_count=4,
                use_cropbox=True,  # Use cropbox for better region detection
                first_page=1,
                last_page=None
            )
            
            if debug:
                print(f"Converting {len(images)} pages to images for OCR")
            
            # Process each page with OCR
            text_content = []
            for idx, image in enumerate(images, 1):
                if debug:
                    print(f"Processing page {idx} with OCR...")
                text = _extract_text_from_image(image)
                if text.strip():  # Only add non-empty text
                    text_content.append(text)
                
        except Exception as e:
            print(f"OCR processing error: {e}")
    
    combined_text = "\n".join(text_content)
    
    if debug:
        print(f"Total extracted text length: {len(combined_text)}")
        print("\nSample of final text:")
        print(combined_text[:500])
        
        # Save debug output
        debug_dir = "debug_ocr"
        os.makedirs(debug_dir, exist_ok=True)
        with open(f"{debug_dir}/extracted_text.txt", "w") as f:
            f.write(combined_text)
    
    return combined_text

def _clean_text(text: str) -> str:
    """Clean and normalize extracted text with enhanced processing."""
    # Convert to lowercase for consistent processing
    text = text.lower()
    
    # Replace common OCR mistakes
    replacements = {
        'l/l': '1/1',  # Common OCR mistake for unit types
        'l/2': '1/2',
        '2/l': '2/1',
        'sq,ft': 'sq ft',
        'sq.ft': 'sq ft',
        'sqft': 'sq ft',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Keep more special characters that might be relevant
    text = re.sub(r'[^\w\s.,;:$%()/-]', '', text)
    
    # Normalize unit types
    text = re.sub(r'(\d)\s*/\s*(\d)', r'\1/\2', text)
    
    # Normalize currency
    text = re.sub(r'\$\s*(\d)', r'$\1', text)
    
    # Normalize percentages
    text = re.sub(r'(\d)\s*%', r'\1%', text)
    
    return text.strip()

def _call_llm(raw_text: str) -> Dict[str, Any]:
    """Hit the ChatCompletion endpoint with enhanced prompt."""
    # Clean the text before sending to OpenAI
    cleaned_text = _clean_text(raw_text)
    
    # Enhanced prompt with more context
    user_prompt = (
        f"{SCHEMA_EXAMPLE}\n\n"
        "Below is the offering memorandum text extracted via OCR. "
        "Note that some numbers might need to be interpreted from context. "
        "Look for both numeric and written forms of numbers. "
        "---\n"
        f"{cleaned_text}\n"
        "---\n\n"
        "Extract the JSON as per the schema. For any fields where you're not completely "
        "confident about the value, use null rather than guessing. JSON only."
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
    
    if not text.strip():
        raise ValueError("No text could be extracted from the document")

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
    p.add_argument("--debug", action="store_true", help="Show debug information")
    args = p.parse_args()

    src = Path(args.source)
    if src.suffix.lower() == ".pdf":
        payload = parse_deal(pdf_path=src, save_json_to=args.out)
    else:
        payload = parse_deal(plain_text=src.read_text(), save_json_to=args.out)

    print(json.dumps(payload, indent=2))
