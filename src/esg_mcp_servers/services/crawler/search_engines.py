"""
Search engine functions for ESG report discovery.

Provides Selenium-based helpers to query Google, Bing and DuckDuckGo and
collect PDF link candidates from the result pages.
"""

import re
import time
import logging
from typing import List, Dict
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants shared with web_crawler.py
# ---------------------------------------------------------------------------

ESG_KEYWORDS = [
    'esg', 'sustainability', 'sustainable', 'environmental', 'social', 'governance',
    'climate', 'carbon', 'emissions', 'greenhouse', 'csr', 'corporate responsibility',
    'non-financial', 'nonfinancial', 'integrated report', 'gri',
]

NEGATIVE_KEYWORDS = [
    'presentation', 'brochure', 'factsheet', 'fact sheet', 'newsletter',
    'product', 'pricing', 'quarterly', 'q1', 'q2', 'q3', 'q4',
]

YEAR_PATTERN = re.compile(r'(20[1-2][0-9])')  # Years 2010-2029


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_pdf_links_from_page_source(page_source: str, base_url: str) -> List[Dict[str, str]]:
    """
    Parse rendered page source and return a list of dicts describing PDF links.

    Each dict contains:
        url   - absolute URL of the PDF
        text  - visible anchor text
        title - title attribute of the anchor (may be empty)
    """
    soup = BeautifulSoup(page_source, 'html.parser')
    results: List[Dict[str, str]] = []
    seen: set = set()

    for anchor in soup.find_all('a', href=True):
        href = anchor['href'].strip()
        if not href:
            continue

        # Resolve relative URLs if a base is available
        if base_url:
            full_url = urljoin(base_url, href)
        else:
            full_url = href

        # We are only interested in links that look like PDFs at this stage;
        # non-PDF result page links that *mention* a PDF are handled by
        # _extract_result_links_from_search_page below.
        lower_url = full_url.lower()
        if '.pdf' not in lower_url:
            continue

        if full_url in seen:
            continue
        seen.add(full_url)

        results.append({
            'url': full_url,
            'text': anchor.get_text(strip=True),
            'title': anchor.get('title', ''),
        })

    return results


def _extract_result_links_from_search_page(page_source: str, num_results: int) -> List[Dict[str, str]]:
    """
    Extract all search-result links from a rendered search engine page.
    Returns dicts with url / text / title keys.
    Only http/https links are returned; javascript: and # anchors are skipped.
    """
    soup = BeautifulSoup(page_source, 'html.parser')
    results: List[Dict[str, str]] = []
    seen: set = set()

    for anchor in soup.find_all('a', href=True):
        href = anchor['href'].strip()

        # Skip non-navigable hrefs
        if not href or href.startswith('#') or href.startswith('javascript:'):
            continue

        # Skip relative paths that are search engine UI elements
        if not href.startswith('http'):
            continue

        if href in seen:
            continue
        seen.add(href)

        results.append({
            'url': href,
            'text': anchor.get_text(strip=True),
            'title': anchor.get('title', ''),
        })

        if len(results) >= num_results * 3:  # gather a generous pool
            break

    return results


def _wait_for_results(driver, timeout: int = 10):
    """Wait until at least one anchor tag is present on the page."""
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.TAG_NAME, 'a'))
        )
    except TimeoutException:
        logger.warning("Timed out waiting for search results to load")


# ---------------------------------------------------------------------------
# Public search-engine functions
# ---------------------------------------------------------------------------

def search_with_google(driver, query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """
    Submit *query* to Google and return a list of PDF link candidates found
    in the search-result page.

    Args:
        driver:      Selenium WebDriver instance.
        query:       Search query string (may include filetype:pdf operators).
        num_results: Maximum number of result links to collect per page.

    Returns:
        List of dicts with keys ``url``, ``text``, ``title``.
    """
    search_url = f"https://www.google.com/search?q={requests.utils.quote(query)}&num={num_results}"
    logger.info(f"[Google] Searching: {search_url}")

    try:
        driver.get(search_url)
        _wait_for_results(driver, timeout=15)
        time.sleep(2)  # allow JS to settle

        page_source = driver.page_source

        # Collect direct PDF links present in the result page markup
        pdf_links = _extract_pdf_links_from_page_source(page_source, search_url)

        # Also collect all result URLs (some may redirect to PDFs)
        result_links = _extract_result_links_from_search_page(page_source, num_results)

        # Merge: prefer direct PDF links, then append other results
        seen_urls = {item['url'] for item in pdf_links}
        for link in result_links:
            if link['url'] not in seen_urls:
                pdf_links.append(link)
                seen_urls.add(link['url'])
            if len(pdf_links) >= num_results * 3:
                break

        logger.info(f"[Google] Collected {len(pdf_links)} candidate links")
        return pdf_links

    except WebDriverException as exc:
        logger.error(f"[Google] WebDriver error: {exc}")
        return []
    except Exception as exc:
        logger.error(f"[Google] Unexpected error: {exc}")
        return []


def search_with_bing(driver, query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """
    Submit *query* to Bing and return a list of PDF link candidates found
    in the search-result page.

    Args:
        driver:      Selenium WebDriver instance.
        query:       Search query string (may include filetype:pdf operators).
        num_results: Maximum number of result links to collect per page.

    Returns:
        List of dicts with keys ``url``, ``text``, ``title``.
    """
    search_url = f"https://www.bing.com/search?q={requests.utils.quote(query)}&count={num_results}"
    logger.info(f"[Bing] Searching: {search_url}")

    try:
        driver.get(search_url)
        _wait_for_results(driver, timeout=15)
        time.sleep(2)

        page_source = driver.page_source

        pdf_links = _extract_pdf_links_from_page_source(page_source, search_url)
        result_links = _extract_result_links_from_search_page(page_source, num_results)

        seen_urls = {item['url'] for item in pdf_links}
        for link in result_links:
            if link['url'] not in seen_urls:
                pdf_links.append(link)
                seen_urls.add(link['url'])
            if len(pdf_links) >= num_results * 3:
                break

        logger.info(f"[Bing] Collected {len(pdf_links)} candidate links")
        return pdf_links

    except WebDriverException as exc:
        logger.error(f"[Bing] WebDriver error: {exc}")
        return []
    except Exception as exc:
        logger.error(f"[Bing] Unexpected error: {exc}")
        return []


def search_with_duckduckgo(driver, query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """
    Submit *query* to DuckDuckGo and return a list of PDF link candidates
    found in the search-result page.

    Args:
        driver:      Selenium WebDriver instance.
        query:       Search query string (may include filetype:pdf operators).
        num_results: Maximum number of result links to collect per page.

    Returns:
        List of dicts with keys ``url``, ``text``, ``title``.
    """
    search_url = f"https://duckduckgo.com/?q={requests.utils.quote(query)}&ia=web"
    logger.info(f"[DuckDuckGo] Searching: {search_url}")

    try:
        driver.get(search_url)
        _wait_for_results(driver, timeout=15)
        time.sleep(3)  # DDG renders results client-side; give it extra time

        page_source = driver.page_source

        pdf_links = _extract_pdf_links_from_page_source(page_source, search_url)
        result_links = _extract_result_links_from_search_page(page_source, num_results)

        seen_urls = {item['url'] for item in pdf_links}
        for link in result_links:
            if link['url'] not in seen_urls:
                pdf_links.append(link)
                seen_urls.add(link['url'])
            if len(pdf_links) >= num_results * 3:
                break

        logger.info(f"[DuckDuckGo] Collected {len(pdf_links)} candidate links")
        return pdf_links

    except WebDriverException as exc:
        logger.error(f"[DuckDuckGo] WebDriver error: {exc}")
        return []
    except Exception as exc:
        logger.error(f"[DuckDuckGo] Unexpected error: {exc}")
        return []
