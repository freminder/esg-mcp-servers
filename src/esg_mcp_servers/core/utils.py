"""Shared utility functions — URL/filename helpers."""
import os
import re
from urllib.parse import urlparse


def get_company_name_and_folder(base_url: str) -> tuple[str, str]:
    """Parse a URL into (company_name, folder_name)."""
    folder_name = base_url.split("//")[-1].split("/")[0]
    domain_parts = folder_name.split(".")

    if len(domain_parts) > 2:
        company_name = domain_parts[-2]
    elif len(domain_parts) == 1:
        match = re.search(r"https?(\w+)com", domain_parts[0])
        company_name = match.group(1) if match else domain_parts[0]
    else:
        company_name = domain_parts[0]

    return company_name, folder_name


def sanitize_filename(filename: str) -> str:
    """Remove filesystem-invalid characters from a filename."""
    return re.sub(r'[<>:"/\\|?*]', "_", filename)


clean_filename = sanitize_filename  # alias


def is_file_exists(file_path: str) -> bool:
    return os.path.exists(file_path)


def extract_company_name(folder_name: str) -> str:
    """Extract company name from a folder name or URL fragment."""
    return re.sub(r"https?://(www\.)?", "", folder_name).split(".")[0]


def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


def get_folder_name(url: str) -> str:
    """Convert a URL into a safe folder name."""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc
        if netloc.startswith("www."):
            netloc = netloc[4:]
        parts = netloc.split(".")
        domain = ".".join(parts[:-2]) if len(parts) > 2 else netloc
        return domain.replace(".", "_")
    except (ValueError, IndexError):
        return "default_folder"


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist; return the path."""
    os.makedirs(path, exist_ok=True)
    return path
