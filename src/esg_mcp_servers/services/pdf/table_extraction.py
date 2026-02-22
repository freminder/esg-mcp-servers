"""PDF table extraction — migrated from esgradar/table_extraction.py.

Hybrid approach: text-pattern matching first, OCR fallback for image-based tables.
Supports multi-domain ESG table classification (emissions, energy, water, waste,
social, governance) per ESRS standards.
"""
import logging
import os
import re
from typing import Any

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract

from pdf2image import convert_from_path
from PIL import Image

logger = logging.getLogger(__name__)

# ── Domain keyword sets for table classification ─────────────────────────────

TABLE_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "emissions": [
        "emission", "carbon", "co2", "ghg", "scope", "tco2e", "greenhouse",
    ],
    "energy": [
        "energy", "electricity", "mwh", "gwh", "kwh", "renewable",
        "non-renewable", "power consumption", "energy intensity",
    ],
    "water": [
        "water", "withdrawal", "discharge", "freshwater", "m3",
        "megalit", "water consumption", "water stress",
    ],
    "waste": [
        "waste", "landfill", "recycl", "hazardous", "non-hazardous",
        "disposal", "circular", "recover",
    ],
    "social": [
        "employee", "headcount", "workforce", "diversity", "gender",
        "injury", "fatality", "training hours", "ltir", "trir",
        "pay gap", "fte", "full-time",
    ],
    "governance": [
        "board", "director", "independence", "committee", "audit",
        "whistleblower", "anti-corruption", "compliance",
    ],
}

# Combined keyword set for OCR detection (any ESG table, not just emissions)
_ALL_ESG_KEYWORDS_RE = re.compile(
    r'(?:emission|carbon|CO2|GHG|Scope|tCO2e|energy|electricity|MWh|GWh'
    r'|water|withdrawal|discharge|waste|landfill|recycl|hazardous'
    r'|employee|headcount|workforce|diversity|injury|fatality'
    r'|board|director|committee|audit|whistleblower)',
    re.IGNORECASE,
)


def _extract_table_text(table: dict) -> str:
    """Flatten all text from a table's header and rows into a single string."""
    data = table.get("data") or {}
    if not data:
        return ""
    parts = []
    if data.get("header"):
        parts.append(" ".join(str(h) for h in data["header"]))
    for row in (data.get("rows") or [])[:5]:
        parts.append(" ".join(str(c) for c in row))
    return " ".join(parts).lower()


def classify_table(table: dict) -> str | None:
    """Classify a table into an ESG domain. Returns domain string or None."""
    text = _extract_table_text(table)
    if not text:
        return None

    scores: dict[str, int] = {}
    for domain, keywords in TABLE_DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[domain] = score

    if not scores:
        return None

    return max(scores, key=scores.get)


# ── Resolve tool paths ──────
_POPPLER_CANDIDATES = [
    os.environ.get("POPPLER_PATH", ""),
    r"C:\Program Files\poppler\Library\bin",
]
POPPLER_PATH: str | None = None
for _p in _POPPLER_CANDIDATES:
    if _p and os.path.isfile(os.path.join(_p, "pdftoppm.exe" if os.name == "nt" else "pdftoppm")):
        POPPLER_PATH = _p
        break

_TESSERACT_CANDIDATES = [
    os.environ.get("TESSERACT_PATH", ""),
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
]
for _t in _TESSERACT_CANDIDATES:
    if _t and os.path.isfile(_t):
        pytesseract.pytesseract.tesseract_cmd = _t
        break

# Function to extract tables from PDF using a combination of techniques
def extract_tables_from_pdf(pdf_path):
    """Extract tables from PDF file using multiple techniques"""
    tables = []

    try:
        # Method 1: Extract text with PyMuPDF and look for table patterns
        doc = fitz.open(pdf_path)
        pages = [(i, doc[i].get_text()) for i in range(len(doc))]
        doc.close()

        # Look for table-like structures in text
        for page_num, content in pages:

            # Look for table markers like rows with multiple numbers
            # Particularly focusing on emissions data patterns
            emission_patterns = [
                # Pattern for CO2 emissions with years and values
                r'(?:19|20)\d{2}[\s\|,]*\d{1,3}(?:,\d{3})*(?:\.\d+)?',
                # Pattern for rows with multiple numbers (likely table)
                r'(\d+(?:,\d{3})*(?:\.\d+)?)[^\d]+(\d+(?:,\d{3})*(?:\.\d+)?)',
                # Pattern for "Scope 1" or "Scope 2" with numbers
                r'Scope\s*[123].*?(\d+(?:,\d{3})*(?:\.\d+)?)',
                # Pattern for year ranges commonly used in emissions reporting
                r'((?:19|20)\d{2})-((?:19|20)\d{2})'
            ]

            # Extract potential table data
            for pattern in emission_patterns:
                matches = re.findall(pattern, content)
                if matches and len(matches) > 3:  # Likely a data table
                    # Try to reconstruct the table structure
                    table_data = extract_structured_data(content, pattern)
                    if table_data and len(table_data) > 0:
                        tables.append({
                            'page': page_num,
                            'data': table_data,
                            'source': 'text_pattern',
                            'confidence': 0.7
                        })

        # Method 2: OCR-based extraction for tables that might be images
        if not tables or len(tables) < 2:  # If first method didn't find much
            tables.extend(extract_tables_via_ocr(pdf_path))

        # Process and classify all extracted tables
        processed_tables = []
        for table in tables:
            domain = classify_table(table)
            if domain:
                cleaned_table = clean_esg_table(table, domain)
                processed_tables.append(cleaned_table)

        domain_counts = {}
        for t in processed_tables:
            d = t.get("table_type", "unknown")
            domain_counts[d] = domain_counts.get(d, 0) + 1
        logger.info(f"Extracted {len(processed_tables)} ESG tables from PDF: {domain_counts}")
        return processed_tables

    except Exception as e:
        logger.error(f"Error extracting tables: {str(e)}")
        return []

def extract_tables_via_ocr(pdf_path):
    """Extract tables using OCR for image-based tables"""
    tables = []

    try:
        # Convert PDF pages to images (poppler_path needed on Windows)
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)

        for i, img in enumerate(images):
            # Convert to OpenCV format
            open_cv_image = np.array(img)
            open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB to BGR

            # Detect table structures using edge detection and contour analysis
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Find table-like structures (rectangles)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours to find potential tables
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Potential table if it's wide enough and tall enough
                if w > img.width * 0.3 and h > img.height * 0.05:
                    # Extract the region and apply OCR
                    roi = open_cv_image[y:y+h, x:x+w]
                    text = pytesseract.image_to_string(roi)

                    # If text contains ESG keywords, it's likely a table we want
                    if _ALL_ESG_KEYWORDS_RE.search(text):
                        structured_data = text_to_structured_data(text)
                        tables.append({
                            'page': i+1,
                            'data': structured_data,
                            'source': 'ocr',
                            'bbox': (x, y, w, h),
                            'confidence': 0.8
                        })

        return tables

    except Exception as e:
        logger.error(f"OCR table extraction error: {str(e)}")
        return []

def extract_structured_data(text_content, pattern):
    """Convert pattern matches into structured table data"""
    # Implement logic to convert raw text to structured data
    # This is a simplified implementation - would need customization
    lines = text_content.split('\n')
    table_data = []

    in_table = False
    header_row = []

    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue

        # Detect potential header row
        if re.search(r'(year|period|20\d{2}|19\d{2})', line, re.IGNORECASE) and not in_table:
            # Try to extract header cells
            cells = re.split(r'\s{2,}|\t|\|', line)
            if len(cells) >= 2:  # At least two columns
                header_row = [cell.strip() for cell in cells if cell.strip()]
                in_table = True
                continue

        # If we're in a table, try to extract row data
        if in_table:
            # Check if line has multiple numbers matching the pattern
            if re.search(pattern, line):
                # Split the line into cells
                cells = re.split(r'\s{2,}|\t|\|', line)
                if len(cells) >= 2:  # At least two columns
                    row_data = [cell.strip() for cell in cells if cell.strip()]
                    # If we have data with at least one number, add it
                    if any(re.search(r'\d', cell) for cell in row_data):
                        table_data.append(row_data)
                else:
                    # Might be end of table if line format changes
                    in_table = False

    # If we found a header and data rows, return as structured data
    if header_row and table_data:
        # Normalize column count
        max_cols = max(len(header_row), max(len(row) for row in table_data))
        normalized_header = header_row + [''] * (max_cols - len(header_row))

        normalized_data = []
        for row in table_data:
            normalized_data.append(row + [''] * (max_cols - len(row)))

        return {'header': normalized_header, 'rows': normalized_data}
    else:
        return None

def text_to_structured_data(text):
    """Convert OCR text to structured table data"""
    lines = text.split('\n')
    rows = []

    for line in lines:
        if line.strip():
            # Split by whitespace, but preserve multi-word cells that might be quoted
            cells = []
            in_quotes = False
            current_cell = ""

            for char in line:
                if char == '"' or char == "'":
                    in_quotes = not in_quotes
                    current_cell += char
                elif char.isspace() and not in_quotes:
                    if current_cell:
                        cells.append(current_cell.strip())
                        current_cell = ""
                else:
                    current_cell += char

            if current_cell:
                cells.append(current_cell.strip())

            # Only add non-empty rows
            if cells:
                rows.append(cells)

    if len(rows) >= 2:  # Need at least header and one data row
        return {'header': rows[0], 'rows': rows[1:]}
    else:
        return None

def is_emissions_table(table):
    """Determine if a table contains emissions data."""
    return classify_table(table) == "emissions"


def clean_esg_table(table, domain: str):
    """Clean and normalize an ESG table, tagging it with its domain."""
    if 'data' not in table or not table['data']:
        table['table_type'] = domain
        return table

    # Apply the standard cleaning (header normalization, number cleanup)
    cleaned = clean_emissions_table(table)
    cleaned['table_type'] = domain

    # For non-emissions tables, store domain-specific metadata in table_meta
    if domain != "emissions":
        text = _extract_table_text(cleaned)
        meta: dict[str, Any] = {"domain": domain}

        # Extract years from table
        years = sorted(set(re.findall(r'((?:19|20)\d{2})', text)))
        if years:
            meta["years"] = years

        # Detect units based on domain
        if domain == "energy":
            if "gwh" in text:
                meta["units"] = "GWh"
            elif "kwh" in text:
                meta["units"] = "kWh"
            else:
                meta["units"] = "MWh"
        elif domain == "water":
            if "megalit" in text:
                meta["units"] = "ML"
            else:
                meta["units"] = "m3"
        elif domain == "waste":
            meta["units"] = "tonnes"

        cleaned['table_meta'] = meta

    return cleaned


def clean_emissions_table(table):
    """Clean and normalize an emissions table"""
    if 'data' not in table or not table['data']:
        return table

    data = table['data']

    # Standardize header names
    if 'header' in data and data['header']:
        # Look for year columns
        for i, header in enumerate(data['header']):
            # Convert any year range (2020-2021) to individual years
            if re.search(r'(19|20)\d{2}[/-](19|20)\d{2}', str(header)):
                years = re.findall(r'(19|20)\d{2}', str(header))
                data['header'][i] = f"Year {years[0]}"

            # Normalize "Year" headers
            elif re.search(r'(19|20)\d{2}', str(header)):
                year = re.search(r'(19|20)\d{2}', str(header)).group(0)
                data['header'][i] = f"Year {year}"

    # Clean row data - normalize numbers and remove commas
    if 'rows' in data and data['rows']:
        for i, row in enumerate(data['rows']):
            for j, cell in enumerate(row):
                # If it's a numeric cell, standardize format
                if isinstance(cell, str) and re.search(r'\d', cell):
                    # Remove commas from numbers
                    cell = re.sub(r'(\d),(\d)', r'\1\2', cell)
                    # Try to extract just the numeric part if mixed with text
                    if re.search(r'[^\d.\-+]', cell):
                        num_match = re.search(r'[-+]?\d*\.?\d+', cell)
                        if num_match:
                            cell = num_match.group(0)
                    data['rows'][i][j] = cell

    # Add metadata about what kind of emissions data this table contains
    table['emissions_type'] = identify_emissions_type(data)
    return table

def identify_emissions_type(table_data):
    """Identify what type of emissions data this table contains"""
    emissions_type = {
        'scope1': False,
        'scope2': False,
        'scope3': False,
        'years': [],
        'units': 'unknown'
    }

    # Check headers and first few rows for scope information
    all_text = ' '.join(
        ' '.join(str(h) for h in table_data.get('header', [])) +
        ' '.join(' '.join(str(c) for c in row) for row in table_data.get('rows', [])[:3])
    ).lower()

    # Detect scope
    emissions_type['scope1'] = bool(re.search(r'scope\s*1|direct emission', all_text))
    emissions_type['scope2'] = bool(re.search(r'scope\s*2|indirect emission', all_text))
    emissions_type['scope3'] = bool(re.search(r'scope\s*3|other indirect|value chain', all_text))

    # Detect units
    if re.search(r't\s*co2e|tonnes|metric tons', all_text):
        emissions_type['units'] = 'tCO2e'
    elif re.search(r'kg\s*co2e|kilogram', all_text):
        emissions_type['units'] = 'kgCO2e'

    # Detect years
    year_pattern = re.compile(r'(19|20)\d{2}')
    years = year_pattern.findall(all_text)
    if years:
        emissions_type['years'] = sorted(list(set(years)))

    return emissions_type
