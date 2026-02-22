FROM python:3.11-slim

WORKDIR /app

# System deps for PDF processing (pdfplumber, pytesseract)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir .

# Default: run the combined server (all 31 tools)
CMD ["esg-mcp-all"]
