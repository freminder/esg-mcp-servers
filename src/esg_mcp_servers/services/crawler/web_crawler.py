"""
ESG Report Crawler and Downloader
This module provides functionality to crawl websites and download ESG/sustainability reports.
"""

import os
import time
import hashlib
import logging
import random
import re
import threading
import queue
import shutil
import requests
from urllib.parse import urlparse, urljoin, unquote
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple

# Third-party imports
from bs4 import BeautifulSoup
from pymongo import MongoClient
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# Platform imports
from esg_mcp_servers.core.state import state
from esg_mcp_servers.core.log_manager import log_messages
from esg_mcp_servers.core.utils import get_company_name_and_folder, is_file_exists, sanitize_filename
from esg_mcp_servers.settings import settings

# Sibling crawler imports
from esg_mcp_servers.services.crawler.chrome_config import configure_selenium
from esg_mcp_servers.services.crawler.search_engines import (
    search_with_google,
    search_with_bing,
    search_with_duckduckgo,
    ESG_KEYWORDS,
    NEGATIVE_KEYWORDS,
    YEAR_PATTERN,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('esg_crawler')


#######################
# Web Crawler Classes #
#######################

class WebCrawler:
    """
    WebCrawler class for crawling websites and finding PDF links.
    """

    def __init__(self, base_url: str, depth_limit: int = 2, max_pages: int = 100,
                 domain_restrict: bool = True, num_workers: int = 5):
        """
        Initialize WebCrawler.

        Args:
            base_url: Starting URL for crawl
            depth_limit: Maximum depth to crawl
            max_pages: Maximum pages to crawl
            domain_restrict: Whether to restrict to same domain
            num_workers: Number of worker threads
        """
        self.base_url = base_url
        self.base_domain = urlparse(base_url).netloc
        self.depth_limit = depth_limit
        self.max_pages = max_pages
        self.domain_restrict = domain_restrict
        self.num_workers = num_workers

        # Initialize collections
        self.url_queue = queue.Queue()
        self.visited_urls = set()
        self.pdf_links = []

        # Statistics
        self.pages_crawled = 0
        self.pdfs_found = 0

        # Add base URL to queue with depth 0
        self.url_queue.put((base_url, 0))

        # Threading lock
        self.lock = threading.Lock()

    def start(self):
        """Start the crawler with multiple worker threads."""
        logger.info(f"Starting crawl from {self.base_url} with {self.num_workers} workers")

        # Create worker threads
        workers = []
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._crawl_worker)
            worker.daemon = True
            workers.append(worker)
            worker.start()

        # Wait for all tasks to complete or max pages reached
        self.url_queue.join()

        # Return collected PDF links
        logger.info(f"Crawl complete. Found {len(self.pdf_links)} PDF links across {self.pages_crawled} pages")
        return self.pdf_links

    def _crawl_worker(self):
        """Worker thread function for crawling URLs from the queue."""
        while True:
            try:
                # Get URL from queue with 1s timeout
                url, depth = self.url_queue.get(timeout=1)

                # Check if we've reached max pages
                with self.lock:
                    if self.pages_crawled >= self.max_pages:
                        self.url_queue.task_done()
                        continue

                # Process the URL
                self._process_url(url, depth)

                # Mark task as done
                self.url_queue.task_done()

            except queue.Empty:
                # No more URLs or timeout
                break
            except Exception as e:
                logger.error(f"Error in crawler worker: {str(e)}")
                self.url_queue.task_done()

    def _process_url(self, url: str, depth: int):
        """
        Process a URL: extract links and PDF links.

        Args:
            url: URL to process
            depth: Current crawl depth
        """
        # Skip if already visited
        if url in self.visited_urls:
            return

        # Add to visited set
        with self.lock:
            self.visited_urls.add(url)
            self.pages_crawled += 1

        logger.info(f"Crawling ({self.pages_crawled}/{self.max_pages}): {url}")

        try:
            # Get page content
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Failed to retrieve {url}: {response.status_code}")
                return

            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract and process links if not at depth limit
            if depth < self.depth_limit:
                self._extract_links(soup, url, depth + 1)

            # Extract PDF links
            self._extract_pdf_links(soup, url)

        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")

    def _extract_links(self, soup: BeautifulSoup, base_url: str, new_depth: int):
        """
        Extract all links from page for further crawling.

        Args:
            soup: BeautifulSoup object of the page
            base_url: Base URL for resolving relative URLs
            new_depth: Depth for new URLs
        """
        for link in soup.find_all('a', href=True):
            href = link['href']

            # Skip empty links, anchors, javascript, etc.
            if not href or href.startswith('#') or href.startswith('javascript:'):
                continue

            # Resolve relative URL
            full_url = urljoin(base_url, href)

            # Clean URL (remove fragments, query params)
            parsed_url = urlparse(full_url)
            clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"

            # Skip if domain restriction is enabled and domain doesn't match
            if self.domain_restrict and urlparse(clean_url).netloc != self.base_domain:
                continue

            # Skip if already visited
            if clean_url in self.visited_urls:
                continue

            # Add to queue
            self.url_queue.put((clean_url, new_depth))

    def _extract_pdf_links(self, soup: BeautifulSoup, base_url: str):
        """
        Extract PDF links from the page.

        Args:
            soup: BeautifulSoup object of the page
            base_url: Base URL for resolving relative URLs
        """
        # Extract PDF links using href attributes
        for link in soup.find_all('a', href=True):
            href = link['href'].strip()

            # Skip empty links
            if not href:
                continue

            # Check if it's a PDF link
            is_pdf = href.lower().endswith('.pdf') or '.pdf' in href.lower()

            if is_pdf:
                # Resolve relative URL
                full_url = urljoin(base_url, href)

                # Extract link text and title
                link_text = link.get_text(strip=True)
                link_title = link.get('title', '')

                # Avoid duplicates
                if not any(item['url'] == full_url for item in self.pdf_links):
                    with self.lock:
                        self.pdf_links.append({
                            'url': full_url,
                            'text': link_text,
                            'title': link_title,
                            'source_page': base_url,
                        })
                        self.pdfs_found += 1

                        logger.info(f"Found PDF: {full_url}")


###############################
# Search Engine PDF Functions #
###############################

def extract_better_company_name(url: str) -> str:
    """
    Extract a better company name from URL using multiple strategies.

    Args:
        url: Company website URL

    Returns:
        More accurate company name
    """
    try:
        # First try to get company name from the website itself
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Try to extract company name from title tag
                title = soup.find('title')
                if title:
                    title_text = title.get_text().strip()
                    # Clean up common title patterns
                    title_text = re.sub(r'\s*[-|--]\s*(Home|Homepage|Official Website|Website).*$', '', title_text, flags=re.IGNORECASE)
                    title_text = re.sub(r'\s*\|\s*.*$', '', title_text)  # Remove everything after |
                    if title_text and len(title_text) > 2 and len(title_text) < 100:
                        logger.info(f"Extracted company name from title: {title_text}")
                        return title_text.strip()

                # Try to extract from meta description or company name
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc and meta_desc.get('content'):
                    desc = meta_desc['content'].strip()
                    # Look for company name patterns in description
                    company_patterns = [
                        r'^([A-Z][a-zA-Z\s&\.]+?)\s+(?:is|offers|provides|specializes)',
                        r'^([A-Z][a-zA-Z\s&\.]+?)\s*[-–]',
                        r'^([A-Z][a-zA-Z\s&\.]+?)\s*\|'
                    ]
                    for pattern in company_patterns:
                        match = re.search(pattern, desc)
                        if match:
                            company_name = match.group(1).strip()
                            if len(company_name) > 2 and len(company_name) < 50:
                                logger.info(f"Extracted company name from meta description: {company_name}")
                                return company_name

        except Exception as e:
            logger.debug(f"Could not extract company name from website content: {e}")

    except Exception as e:
        logger.debug(f"Error in extract_better_company_name: {e}")

    # Fallback to domain-based extraction with improvements
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()

    # Remove common prefixes
    domain = re.sub(r'^(www\.|m\.|mobile\.)', '', domain)

    # Split domain and get the main part
    parts = domain.split('.')
    if len(parts) >= 2:
        main_part = parts[0]

        # Handle special cases and known patterns
        special_cases = {
            'realitymaps': 'Reality Maps',
            '3dresearch': '3D Research',
            '3dwave': '3D Wave',
            '3d4mec': '3D4MEC',
        }

        if main_part in special_cases:
            return special_cases[main_part]

        # Convert camelCase or handle numbers
        if re.search(r'[0-9]', main_part):
            # Handle cases with numbers
            main_part = re.sub(r'([a-z])([0-9])', r'\1 \2', main_part)
            main_part = re.sub(r'([0-9])([a-z])', r'\1 \2', main_part)

        # Convert to title case
        company_name = main_part.replace('-', ' ').replace('_', ' ').title()

        return company_name

    return "Unknown Company"


def filter_esg_reports(pdf_links: List[Dict[str, str]], company_name: str, original_url: str = "") -> List[Dict[str, str]]:
    """
    Filter PDF links to include only ESG/sustainability related reports with improved company matching.

    Args:
        pdf_links: List of dictionaries containing PDF link information
        company_name: Name of the company for additional filtering
        original_url: Original company URL for better filtering

    Returns:
        List of filtered PDF links
    """
    # Create more sophisticated company name variations
    company_variations = []

    # Original company name
    company_variations.append(company_name.lower())

    # Remove common corporate suffixes
    clean_name = re.sub(r'\s+(inc|ltd|llc|corp|corporation|company|co|ag|gmbh|sa|plc)\.?$', '', company_name.lower())
    if clean_name != company_name.lower():
        company_variations.append(clean_name)

    # Add space-removed version
    company_variations.append(company_name.lower().replace(' ', ''))

    # Add abbreviation
    words = company_name.split()
    if len(words) > 1:
        abbreviation = ''.join(word[0] for word in words if word)
        company_variations.append(abbreviation.lower())

    # Extract domain for additional validation
    domain_part = ""
    if original_url:
        try:
            parsed_url = urlparse(original_url)
            domain_part = parsed_url.netloc.lower().replace('www.', '').split('.')[0]
            company_variations.append(domain_part)
        except:
            pass

    # Remove duplicates and empty strings
    company_variations = list(set([var for var in company_variations if var and len(var) > 1]))

    logger.info(f"Company variations for filtering: {company_variations}")

    # Regular expression patterns
    esg_pattern = re.compile('|'.join(ESG_KEYWORDS), re.IGNORECASE)
    negative_pattern = re.compile('|'.join(NEGATIVE_KEYWORDS), re.IGNORECASE)

    filtered_links = []

    for link in pdf_links:
        url = link.get('url', '').lower()
        title = link.get('title', '').lower()
        text = link.get('text', '').lower()

        # Skip if URL doesn't end with .pdf
        if not url.endswith('.pdf'):
            continue

        # Extract filename from URL for additional checking
        filename = os.path.basename(urlparse(url).path).lower()
        filename = unquote(filename)  # URL decode

        # Combined text for searching
        combined_text = f"{title} {text} {filename} {url}"

        # Check for ESG keywords
        if not esg_pattern.search(combined_text):
            continue

        # Check if it's likely a report (contains year)
        if not YEAR_PATTERN.search(combined_text):
            continue

        # Improved company name matching with scoring
        company_match_score = 0
        total_variations = len(company_variations)

        for variation in company_variations:
            if variation in combined_text:
                # Give higher score for exact matches in URL domain
                if variation in url and 'http' in url:
                    company_match_score += 3
                # Medium score for matches in title
                elif variation in title:
                    company_match_score += 2
                # Lower score for matches in text/description
                else:
                    company_match_score += 1

        # Require at least one strong match or multiple weak matches
        if company_match_score < 1:
            logger.debug(f"Skipping PDF - no company match: {filename}")
            continue

        # Additional validation: check if the URL domain matches our target company
        if original_url and domain_part:
            link_domain = ""
            try:
                link_parsed = urlparse(url)
                link_domain = link_parsed.netloc.lower().replace('www.', '').split('.')[0]
            except:
                pass

            # If domains are completely different and we don't have a strong company match, skip
            if link_domain and link_domain != domain_part and company_match_score < 2:
                logger.debug(f"Skipping PDF - domain mismatch: {link_domain} vs {domain_part}, score: {company_match_score}")
                continue

        # Skip if negative keywords are present
        if negative_pattern.search(title) or negative_pattern.search(filename):
            continue

        # Add match score to the link for debugging
        link['match_score'] = company_match_score
        filtered_links.append(link)
        logger.info(f"Including PDF (score: {company_match_score}): {filename}")

    return filtered_links


def search_and_download_esg_reports(company_name: str, download_folder: str = None,
                                    search_limit: int = 20, save_to_mongo: bool = True,
                                    original_url: str = "") -> List[str]:
    """
    Search for ESG/sustainability reports using search engines and download them.

    Args:
        company_name: Name of the company to search for
        download_folder: Folder to save downloaded reports
        search_limit: Maximum number of search results to process
        save_to_mongo: Whether to save files to MongoDB automatically
        original_url: Original company URL for better filtering

    Returns:
        List of downloaded file paths
    """
    # Create company-specific folder if download_folder is not specified
    if download_folder is None:
        download_folder = os.path.join('downloads', sanitize_filename(company_name))

    os.makedirs(download_folder, exist_ok=True)
    downloaded_files = []

    # Set up more specific search queries with company name validation
    search_queries = [
        f'"{company_name}" sustainability report filetype:pdf',
        f'"{company_name}" ESG report filetype:pdf',
        f'"{company_name}" annual report ESG filetype:pdf',
        f'"{company_name}" corporate responsibility report filetype:pdf',
        f'"{company_name}" environmental report filetype:pdf',
    ]

    # Add domain-specific search if we have the URL
    if original_url:
        try:
            domain = urlparse(original_url).netloc.replace('www.', '')
            search_queries.insert(0, f'site:{domain} sustainability OR ESG filetype:pdf')
            search_queries.insert(1, f'site:{domain} "sustainability report" OR "ESG report" filetype:pdf')
        except:
            pass

    driver = None
    try:
        logger.info("Initializing Chrome WebDriver...")
        driver = configure_selenium(download_folder)
        logger.info("Chrome WebDriver initialized successfully")

        search_engines = [search_with_google, search_with_bing, search_with_duckduckgo]

        # Process each search query
        for query_idx, search_query in enumerate(search_queries):
            logger.info(f"Searching for: {search_query}")

            # Try each search engine
            for engine_idx, search_engine in enumerate(search_engines):
                try:
                    pdf_links = search_engine(driver, search_query, search_limit)

                    if pdf_links:
                        logger.info(f"Found {len(pdf_links)} potential PDF links from {search_engine.__name__}")

                        # Filter for ESG reports with improved company matching
                        esg_pdf_links = filter_esg_reports(pdf_links, company_name, original_url)
                        logger.info(f"Filtered to {len(esg_pdf_links)} relevant ESG reports")

                        # Sort by match score (highest first)
                        esg_pdf_links.sort(key=lambda x: x.get('match_score', 0), reverse=True)

                        # Download filtered PDFs
                        for link_idx, link in enumerate(esg_pdf_links):
                            try:
                                file_path = download_pdf(link['url'], download_folder, company_name)
                                if file_path:
                                    downloaded_files.append(file_path)
                                    logger.info(f"Downloaded: {os.path.basename(file_path)}")

                                    # Add delay between downloads
                                    time.sleep(random.uniform(1, 3))

                            except Exception as e:
                                logger.error(f"Error downloading {link['url']}: {str(e)}")

                    # Add delay between search engines
                    time.sleep(random.uniform(3, 5))

                except Exception as e:
                    logger.error(f"Error with {search_engine.__name__}: {str(e)}")

        # Save to MongoDB if requested and files were downloaded
        if save_to_mongo and downloaded_files:
            try:
                logger.info(f"Attempting to save {len(downloaded_files)} files to MongoDB for {company_name}")
                saved_count = save_files_to_mongodb(downloaded_files, company_name)
                logger.info(f"Successfully saved {saved_count} files to MongoDB for {company_name}")

                # Additional verification
                if saved_count != len(downloaded_files):
                    logger.warning(f"Mismatch: Downloaded {len(downloaded_files)} files but saved {saved_count} to MongoDB")

            except Exception as e:
                logger.error(f"Error saving files to MongoDB: {str(e)}")

        return downloaded_files

    except Exception as e:
        logger.error(f"Error initializing WebDriver: {str(e)}")
        return downloaded_files

    finally:
        if driver is not None:
            logger.info("Closing Chrome WebDriver")
            try:
                driver.quit()
            except Exception as e:
                logger.error(f"Error closing Chrome WebDriver: {str(e)}")


def search_and_download_esg_reports_with_cancellation(company_name: str, download_folder: str = None,
                                                     search_limit: int = 20, save_to_mongo: bool = True,
                                                     original_url: str = "", cancellation_check=None) -> List[str]:
    """
    Search for ESG/sustainability reports with cancellation support for batch operations.

    Args:
        company_name: Name of the company to search for
        download_folder: Folder to save downloaded reports
        search_limit: Maximum number of search results to process
        save_to_mongo: Whether to save files to MongoDB automatically
        original_url: Original company URL for better filtering
        cancellation_check: Function that returns True if operation should be cancelled

    Returns:
        List of downloaded file paths
    """
    # Create company-specific folder if download_folder is not specified
    if download_folder is None:
        download_folder = os.path.join('downloads', sanitize_filename(company_name))

    os.makedirs(download_folder, exist_ok=True)
    downloaded_files = []

    # Check for cancellation before starting
    if cancellation_check and cancellation_check():
        logger.info(f"Operation cancelled before starting search for {company_name}")
        return downloaded_files

    # Set up more effective search queries
    search_queries = []

    # Add domain-specific search if we have the URL (but make it more flexible)
    if original_url:
        try:
            domain = urlparse(original_url).netloc.replace('www.', '')
            # Only add site-specific search for well-known domains
            if any(tld in domain for tld in ['.com', '.org', '.net', '.co.', '.de', '.fr', '.uk']):
                # First try exact site search
                search_queries.append(f'site:{domain} sustainability OR ESG filetype:pdf')
                search_queries.append(f'site:{domain} "sustainability report" OR "ESG report" filetype:pdf')

            # Also try searching with the domain name as a keyword
            domain_name = domain.split('.')[0]
            if len(domain_name) > 3:  # Only if domain name is meaningful
                search_queries.append(f'"{domain_name}" OR "{company_name}" sustainability report filetype:pdf')
        except:
            pass

    # Add company-specific searches (these are most likely to work)
    search_queries.extend([
        f'"{company_name}" sustainability report filetype:pdf',
        f'"{company_name}" ESG report filetype:pdf',
        f'"{company_name}" annual report ESG filetype:pdf',
        f'"{company_name}" corporate responsibility report filetype:pdf',
        f'"{company_name}" environmental report filetype:pdf',
    ])

    # Add broader searches without quotes for companies with unusual names
    if any(char in company_name for char in ['3', '4', 'digital', 'tech', 'map']):
        search_queries.extend([
            f'{company_name} sustainability report filetype:pdf',
            f'{company_name} ESG report filetype:pdf',
            f'{company_name.replace(" ", "")} sustainability filetype:pdf',
        ])

    driver = None
    try:
        logger.info("Initializing Chrome WebDriver...")
        driver = configure_selenium(download_folder)
        logger.info("Chrome WebDriver initialized successfully")

        search_engines = [search_with_google, search_with_bing, search_with_duckduckgo]

        # Process each search query
        for query_idx, search_query in enumerate(search_queries):
            # Check for cancellation before each query
            if cancellation_check and cancellation_check():
                logger.info(f"Operation cancelled during query {query_idx + 1}/{len(search_queries)} for {company_name}")
                break

            logger.info(f"Searching for: {search_query}")

            # Try each search engine
            for engine_idx, search_engine in enumerate(search_engines):
                # Check for cancellation before each search engine
                if cancellation_check and cancellation_check():
                    logger.info(f"Operation cancelled during search engine {engine_idx + 1}/{len(search_engines)} for {company_name}")
                    break

                try:
                    pdf_links = search_engine(driver, search_query, search_limit)

                    # Check for cancellation after getting results
                    if cancellation_check and cancellation_check():
                        logger.info(f"Operation cancelled after search results for {company_name}")
                        break

                    if pdf_links:
                        logger.info(f"Found {len(pdf_links)} potential PDF links from {search_engine.__name__}")

                        # Filter for ESG reports with improved company matching
                        esg_pdf_links = filter_esg_reports(pdf_links, company_name, original_url)
                        logger.info(f"Filtered to {len(esg_pdf_links)} relevant ESG reports")

                        # Sort by match score (highest first)
                        esg_pdf_links.sort(key=lambda x: x.get('match_score', 0), reverse=True)

                        # Download filtered PDFs
                        for link_idx, link in enumerate(esg_pdf_links):
                            # Check for cancellation before each download
                            if cancellation_check and cancellation_check():
                                logger.info(f"Operation cancelled during download {link_idx + 1}/{len(esg_pdf_links)} for {company_name}")
                                return downloaded_files

                            try:
                                file_path = download_pdf(link['url'], download_folder, company_name)
                                if file_path:
                                    downloaded_files.append(file_path)
                                    logger.info(f"Downloaded: {os.path.basename(file_path)}")

                                    # Add delay between downloads with cancellation check
                                    for i in range(3):  # 3 second delay split into 1-second checks
                                        if cancellation_check and cancellation_check():
                                            logger.info(f"Operation cancelled during download delay for {company_name}")
                                            return downloaded_files
                                        time.sleep(1)

                            except Exception as e:
                                logger.error(f"Error downloading {link['url']}: {str(e)}")

                    # Add delay between search engines with cancellation check
                    for i in range(5):  # 5 second delay split into 1-second checks
                        if cancellation_check and cancellation_check():
                            logger.info(f"Operation cancelled during search engine delay for {company_name}")
                            return downloaded_files
                        time.sleep(1)

                except Exception as e:
                    logger.error(f"Error with {search_engine.__name__}: {str(e)}")

            # If stopped, break out of outer loop too
            if cancellation_check and cancellation_check():
                break

        # Final check before saving to MongoDB
        if cancellation_check and cancellation_check():
            logger.info(f"Operation cancelled before MongoDB save for {company_name}")
            return downloaded_files

        # Save to MongoDB if requested and files were downloaded
        if save_to_mongo and downloaded_files:
            try:
                logger.info(f"Attempting to save {len(downloaded_files)} files to MongoDB for {company_name}")
                saved_count = save_files_to_mongodb(downloaded_files, company_name)
                logger.info(f"Successfully saved {saved_count} files to MongoDB for {company_name}")

                # Additional verification
                if saved_count != len(downloaded_files):
                    logger.warning(f"Mismatch: Downloaded {len(downloaded_files)} files but saved {saved_count} to MongoDB")

            except Exception as e:
                logger.error(f"Error saving files to MongoDB: {str(e)}")

        return downloaded_files

    except Exception as e:
        logger.error(f"Error initializing WebDriver: {str(e)}")
        return downloaded_files

    finally:
        if driver is not None:
            logger.info("Closing Chrome WebDriver")
            try:
                driver.quit()
            except Exception as e:
                logger.error(f"Error closing Chrome WebDriver: {str(e)}")


def download_pdf(url: str, download_folder: str, company_name: str) -> Optional[str]:
    """
    Download a PDF file from a URL.

    Args:
        url: URL of the PDF to download
        download_folder: Folder to save the PDF
        company_name: Company name for filename construction

    Returns:
        Path to downloaded file or None if failed
    """
    try:
        # Add headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf,application/octet-stream,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        response = requests.get(url, stream=True, timeout=30, headers=headers, allow_redirects=True)
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get('Content-Type', '').lower()

        # More flexible PDF detection
        is_pdf = False

        # Method 1: Check Content-Type header
        if 'application/pdf' in content_type:
            is_pdf = True
            logger.info(f"PDF detected by Content-Type: {content_type}")

        # Method 2: Check URL extension
        elif url.lower().endswith('.pdf'):
            is_pdf = True
            logger.info(f"PDF detected by URL extension: {url}")

        # Method 3: Check actual file content (PDF magic bytes)
        elif not is_pdf:
            # Read first few bytes to check for PDF signature
            first_chunk = next(response.iter_content(chunk_size=1024), b'')
            if first_chunk.startswith(b'%PDF-'):
                is_pdf = True
                logger.info(f"PDF detected by magic bytes in content")
                # Reset the response by making a new request
                response = requests.get(url, stream=True, timeout=30, headers=headers, allow_redirects=True)
                response.raise_for_status()
            else:
                # Check if it's HTML content that might contain the actual PDF link
                content_preview = first_chunk.decode('utf-8', errors='ignore').lower()
                if '<html' in content_preview or '<!doctype html' in content_preview:
                    logger.warning(f"Received HTML content instead of PDF for: {url}")
                    logger.warning(f"Content-Type: {content_type}")
                    logger.warning(f"Content preview: {content_preview[:200]}...")

                    # Try to extract a direct PDF link from the HTML
                    try:
                        from bs4 import BeautifulSoup
                        response_full = requests.get(url, timeout=30, headers=headers)
                        soup = BeautifulSoup(response_full.content, 'html.parser')

                        # Look for direct PDF links in the HTML
                        pdf_links = soup.find_all('a', href=True)
                        for link in pdf_links:
                            href = link['href']
                            if href.lower().endswith('.pdf'):
                                full_pdf_url = urljoin(url, href)
                                logger.info(f"Found direct PDF link in HTML: {full_pdf_url}")
                                # Recursively try to download the direct PDF link
                                return download_pdf(full_pdf_url, download_folder, company_name)

                        # Look for meta refresh or JavaScript redirects
                        meta_refresh = soup.find('meta', attrs={'http-equiv': 'refresh'})
                        if meta_refresh and 'content' in meta_refresh.attrs:
                            content = meta_refresh['content']
                            if 'url=' in content.lower():
                                redirect_url = content.split('url=')[1].strip()
                                if redirect_url.lower().endswith('.pdf'):
                                    logger.info(f"Found PDF redirect in meta refresh: {redirect_url}")
                                    return download_pdf(redirect_url, download_folder, company_name)

                    except Exception as e:
                        logger.debug(f"Error parsing HTML for PDF links: {e}")

                    return None

        if not is_pdf:
            logger.warning(f"Not a PDF: {url} (Content-Type: {content_type})")
            return None

        # Extract filename from URL or create one with company name and timestamp
        url_filename = os.path.basename(urlparse(url).path)
        if url_filename.lower().endswith('.pdf'):
            # Clean up the filename
            filename = unquote(url_filename)

            # Add company prefix if not already present
            if company_name.lower() not in filename.lower():
                # Extract year if present
                year_match = YEAR_PATTERN.search(filename)
                year = year_match.group(1) if year_match else ''

                # Create new filename with company and year
                filename = f"{company_name.replace(' ', '_')}_{year}_{filename}"
        else:
            # Create filename with company name and timestamp
            timestamp = int(time.time())
            filename = f"{company_name.replace(' ', '_')}_report_{timestamp}.pdf"

        # Remove invalid characters
        filename = re.sub(r'[\\/*?:"<>|]', '', filename)

        # Save to disk
        file_path = os.path.join(download_folder, filename)
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Verify file size is reasonable (> 10 KB)
        file_size = os.path.getsize(file_path)
        if file_size < 10240:
            logger.warning(f"File too small ({file_size} bytes), might not be a valid PDF: {file_path}")
            os.remove(file_path)
            return None

        # Additional verification: Check if the downloaded file actually contains PDF content
        try:
            with open(file_path, 'rb') as f:
                first_bytes = f.read(10)
                if not first_bytes.startswith(b'%PDF-'):
                    logger.warning(f"Downloaded file doesn't have PDF magic bytes: {file_path}")
                    logger.warning(f"First 10 bytes: {first_bytes}")
                    os.remove(file_path)
                    return None
        except Exception as e:
            logger.warning(f"Error verifying PDF content: {e}")

        logger.info(f"Successfully downloaded PDF: {filename} ({file_size} bytes)")
        return file_path

    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return None


#######################
# MongoDB Operations  #
#######################

def compute_file_hash(file_path: str) -> str:
    """
    Compute MD5 hash of file.

    Args:
        file_path: Path to file

    Returns:
        MD5 hash as hexadecimal string
    """
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def save_files_to_mongodb(file_paths: List[str], company_name: str) -> int:
    """
    Save downloaded files to MongoDB with duplicate checking.

    Args:
        file_paths: List of file paths to save
        company_name: Company name for metadata

    Returns:
        Number of files actually saved (excluding duplicates)
    """
    try:
        import gridfs

        # Connection URI and database name come from centralised settings
        mongodb_uri = settings.MONGODB_URI
        mongodb_db = settings.MONGODB_DB

        # Try multiple MongoDB connection options, preferring the configured URI
        connection_options = [
            mongodb_uri,
            'mongodb://localhost:27017/',
            'mongodb://127.0.0.1:27017/',
            'mongodb://localhost:27017/?directConnection=true',
        ]
        # De-duplicate while preserving order
        seen_uris: set = set()
        unique_uris = []
        for uri in connection_options:
            if uri not in seen_uris:
                seen_uris.add(uri)
                unique_uris.append(uri)

        client = None
        for uri in unique_uris:
            try:
                client = MongoClient(uri, serverSelectionTimeoutMS=5000)
                client.server_info()  # Test connection
                logger.info(f"MongoDB connection successful with URI: {uri}")
                break
            except Exception as e:
                logger.debug(f"Failed to connect with URI {uri}: {str(e)}")
                continue

        if client is None:
            logger.error("Failed to connect to MongoDB with all connection options")
            logger.error("Please ensure MongoDB is running and accessible")
            logger.error("Try starting MongoDB with: mongod --dbpath C:\\data\\db")
            return 0

        db = client[mongodb_db]
        fs = gridfs.GridFS(db)

        saved_count = 0
        logger.info(f"Attempting to save {len(file_paths)} files to MongoDB for company: {company_name}")

        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    logger.warning(f"File does not exist: {file_path}")
                    continue

                filename = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)

                logger.info(f"Processing file: {filename} ({file_size} bytes)")

                # Compute file hash to check for duplicates
                file_hash = compute_file_hash(file_path)
                logger.debug(f"File hash for {filename}: {file_hash}")

                # Check if file already exists by hash (more reliable than filename)
                existing_file = fs.find_one({"metadata.file_hash": file_hash})

                if existing_file:
                    logger.info(f"Duplicate file detected (by hash), skipping: {filename}")
                    continue

                # Also check by filename and company (fallback)
                existing_file_by_name = fs.find_one({
                    "filename": filename,
                    "metadata.company_name": company_name,
                })

                if existing_file_by_name:
                    logger.info(f"Duplicate file detected (by name), skipping: {filename}")
                    continue

                # Read and save file
                with open(file_path, 'rb') as f:
                    file_data = f.read()

                    if len(file_data) == 0:
                        logger.warning(f"File is empty, skipping: {filename}")
                        continue

                    # Save to GridFS with comprehensive metadata
                    file_id = fs.put(
                        file_data,
                        filename=filename,
                        content_type="application/pdf",
                        metadata={
                            "company_name": company_name,
                            "file_hash": file_hash,
                            "file_size": file_size,
                            "upload_date": datetime.now(),
                            "source": "esg_crawler",
                            "original_path": file_path,
                        },
                    )

                    logger.info(f"Successfully saved to MongoDB: {filename} (ID: {file_id}, Size: {file_size} bytes)")
                    saved_count += 1

                    # Verify the file was actually saved
                    verification = fs.find_one({"_id": file_id})
                    if verification:
                        logger.debug(f"Verification successful for {filename}")
                    else:
                        logger.error(f"Verification failed for {filename}")

            except Exception as e:
                logger.error(f"Error saving file {file_path} to MongoDB: {str(e)}")
                continue

        logger.info(f"MongoDB save operation completed. Saved {saved_count} out of {len(file_paths)} files")

        # Final verification - count files for this company
        try:
            total_files_for_company = fs.find({"metadata.company_name": company_name}).count()
            logger.info(f"Total files in MongoDB for {company_name}: {total_files_for_company}")
        except Exception as e:
            logger.warning(f"Could not verify total file count: {str(e)}")

        return saved_count

    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        logger.error("MongoDB troubleshooting steps:")
        logger.error("1. Check if MongoDB is running: tasklist | findstr mongod")
        logger.error("2. Start MongoDB: mongod --dbpath C:\\data\\db")
        logger.error("3. Or run as admin: net start MongoDB")
        return 0


def save_pdf_to_mongodb(folder_path: str) -> int:
    """
    Save all PDF files from a folder to MongoDB using the same logic as save_files_to_mongodb.
    This function provides compatibility with the existing utils.save_pdf_to_mongodb interface.

    Args:
        folder_path: Path to folder containing PDF files

    Returns:
        Number of files saved to MongoDB
    """
    try:
        logger.info(f"Starting MongoDB save operation for folder: {folder_path}")

        if not os.path.exists(folder_path):
            logger.error(f"Folder does not exist: {folder_path}")
            return 0

        # Get all PDF files in the folder
        pdf_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    full_path = os.path.join(root, file)
                    pdf_files.append(full_path)
                    logger.debug(f"Found PDF file: {full_path}")

        if not pdf_files:
            logger.warning(f"No PDF files found in {folder_path}")
            return 0

        logger.info(f"Found {len(pdf_files)} PDF files in {folder_path}")

        # Extract company name from folder path
        folder_name = os.path.basename(folder_path)
        company_name = folder_name.replace('_', ' ').title()
        logger.info(f"Extracted company name: {company_name}")

        # Use the existing save_files_to_mongodb function
        saved_count = save_files_to_mongodb(pdf_files, company_name)

        logger.info(f"MongoDB save operation completed for {company_name}: {saved_count} files saved")

        return saved_count

    except Exception as e:
        logger.error(f"Error in save_pdf_to_mongodb: {str(e)}")
        return 0


###########################
# High-level entry points #
###########################

def run_from_years(company_name: str, years: List[int],
                   base_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Search for ESG reports for specific years.

    Args:
        company_name: Company name to search for
        years: List of years to search for
        base_url: Optional company website to try first

    Returns:
        Dictionary with results
    """
    os.makedirs('downloads', exist_ok=True)
    download_folder = os.path.join('downloads', sanitize_filename(company_name))
    os.makedirs(download_folder, exist_ok=True)

    results: Dict[str, Any] = {
        "company": company_name,
        "years_searched": years,
        "years_found": [],
        "reports": [],
    }

    # Try base URL first if provided
    if base_url:
        logger.info(f"Checking company website first: {base_url}")
        website_result = run_from_url(base_url, download_esg_only=True)

        # If successful, check which years were found
        if website_result.get("success", False):
            # Check downloaded files for years
            for filename in os.listdir(download_folder):
                for year in years:
                    if str(year) in filename:
                        results["years_found"].append(year)
                        results["reports"].append({
                            "year": year,
                            "filename": filename,
                            "source": "company_website",
                        })

    # Search for years not found yet
    missing_years = [year for year in years if year not in results["years_found"]]

    for year in missing_years:
        logger.info(f"Searching for {company_name} ESG report for year {year}")

        search_queries = [
            f"{company_name} sustainability report {year} filetype:pdf",
            f"{company_name} ESG report {year} filetype:pdf",
            f"{company_name} annual report {year} ESG filetype:pdf",
        ]

        # Use search engines to find reports
        temp_folder = f"temp_{company_name}_{year}_{int(time.time())}"
        os.makedirs(temp_folder, exist_ok=True)

        found_report = False
        driver = None

        try:
            driver = configure_selenium(temp_folder)

            # Try each search query
            for search_query in search_queries:
                if found_report:
                    break

                # Try each search engine
                for search_engine in [search_with_google, search_with_bing, search_with_duckduckgo]:
                    if found_report:
                        break

                    try:
                        pdf_links = search_engine(driver, search_query, 10)

                        if pdf_links:
                            # Filter for ESG reports with specific year
                            year_pdf_links = []
                            for link in pdf_links:
                                link_text = link.get('text', '').lower()
                                link_url = link.get('url', '').lower()
                                link_title = link.get('title', '').lower()

                                # Must contain year
                                if str(year) not in f"{link_text} {link_url} {link_title}":
                                    continue

                                # Must be PDF
                                if not link_url.endswith('.pdf'):
                                    continue

                                # Must contain company name
                                if company_name.lower() not in f"{link_text} {link_url} {link_title}".lower():
                                    continue

                                # Add to filtered list
                                year_pdf_links.append(link)

                            # Download the first matching PDF
                            if year_pdf_links:
                                file_path = download_pdf(year_pdf_links[0]['url'], temp_folder, company_name)

                                if file_path:
                                    # Move to main download folder
                                    filename = os.path.basename(file_path)
                                    dest_path = os.path.join(download_folder, filename)

                                    shutil.copy2(file_path, dest_path)

                                    # Add to results
                                    results["years_found"].append(year)
                                    results["reports"].append({
                                        "year": year,
                                        "filename": filename,
                                        "source": f"search_{search_engine.__name__.replace('search_with_', '')}",
                                    })

                                    found_report = True
                                    break
                    except Exception as e:
                        logger.error(f"Error with {search_engine.__name__} for year {year}: {str(e)}")

            # Save to MongoDB using our unified function
            save_pdf_to_mongodb(download_folder)

        except Exception as e:
            logger.error(f"Error searching for year {year}: {str(e)}")

        finally:
            # Clean up
            try:
                if driver is not None:
                    driver.quit()

                if os.path.exists(temp_folder):
                    shutil.rmtree(temp_folder)
            except:
                pass

    # Return final results
    return results


def run_from_url(url: str, download_esg_only: bool = True) -> Dict[str, Any]:
    """
    Crawl website starting from a URL and download ESG reports.

    Args:
        url: The URL to start crawling from
        download_esg_only: If True, only download ESG/sustainability reports

    Returns:
        Dictionary with results
    """
    from services.crawler.pdf_downloader import extract_pdf_links, download_pdfs as http_download_pdfs

    company_name, download_folder = get_company_name_and_folder(url)

    # Special handling for URLs with fragment identifiers
    if "#" in url:
        # Use Selenium approach for JavaScript-heavy pages
        logger.info(f"URL contains fragment identifier, using Selenium for: {url}")

        driver = configure_selenium(download_folder)

        try:
            from downloader import crawl_site, download_pdfs as selenium_download_pdfs

            # Crawl the site and find PDF links
            pdf_links = crawl_site(driver, url)

            if not pdf_links:
                logger.info(f"No PDF links found for URL: {url}")
                return {"success": False, "message": "No PDF links found"}

            # Filter for ESG reports if requested
            if download_esg_only:
                filtered_links = []

                for link in pdf_links:
                    link_text = link.get('text', '').lower() if isinstance(link, dict) else ''
                    link_url = (link.get('url', '') if isinstance(link, dict) else link).lower()

                    # Check if any ESG keyword is in the link text or URL
                    is_esg = any(keyword in link_text or keyword in link_url for keyword in ESG_KEYWORDS)

                    if is_esg:
                        filtered_links.append(link)

                if not filtered_links:
                    logger.info(f"No ESG/sustainability reports found for URL: {url}")
                    return {
                        "success": False,
                        "message": "No ESG reports found",
                        "total_pdfs": len(pdf_links),
                    }

                pdf_links = filtered_links

            # Download the PDFs
            logger.info(f"Found {len(pdf_links)} relevant PDF documents to download")
            selenium_download_pdfs(driver, pdf_links, download_folder)

            # Save to MongoDB using our unified function
            saved_count = save_pdf_to_mongodb(download_folder)
            logger.info(f"Saved {saved_count} files to MongoDB")

            return {
                "success": True,
                "pdfs_found": len(pdf_links),
                "download_folder": download_folder,
                "company": company_name,
            }

        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            return {"success": False, "message": str(e)}

        finally:
            driver.quit()

    # Standard approach for simple URLs
    logger.info(f"Crawling website: {url}")

    # Initialize and start the crawler
    crawler = WebCrawler(url, depth_limit=2, max_pages=50)
    all_pdf_links = crawler.start()

    if not all_pdf_links:
        logger.info(f"No PDF links found for URL: {url}")
        return {"success": False, "message": "No PDF links found"}

    # Filter for ESG reports if requested
    if download_esg_only:
        # Use the improved filter_esg_reports function with 3 parameters
        pdf_links = filter_esg_reports(all_pdf_links, company_name, url)
        if not pdf_links:
            logger.info(f"No ESG/sustainability reports found for URL: {url}")
            return {
                "success": False,
                "message": "No ESG reports found",
                "total_pdfs": len(all_pdf_links),
            }
    else:
        pdf_links = all_pdf_links

    # Download filtered PDFs
    logger.info(f"Downloading {len(pdf_links)} PDF files")
    http_download_pdfs(pdf_links, download_folder)

    # Save to MongoDB using our unified function
    saved_count = save_pdf_to_mongodb(download_folder)
    logger.info(f"Saved {saved_count} files to MongoDB")

    return {
        "success": True,
        "pdfs_found": len(pdf_links),
        "download_folder": download_folder,
        "company": company_name,
    }


def run_from_excel(excel_path: str, sheet_name: str = 'data', url_column: str = 'URL',
                   download_esg_only: bool = True) -> List[Dict[str, Any]]:
    """
    Process URLs from an Excel file.

    Args:
        excel_path: Path to Excel file
        sheet_name: Sheet name containing URLs
        url_column: Column name containing URLs
        download_esg_only: If True, only download ESG/sustainability reports

    Returns:
        List of results for each URL
    """
    try:
        import pandas as pd

        # Load Excel file
        df = pd.read_excel(excel_path, sheet_name=sheet_name)

        if url_column not in df.columns:
            logger.error(f"Column '{url_column}' not found in Excel file")
            return [{"success": False, "message": f"Column '{url_column}' not found"}]

        # Process each URL
        results = []

        for index, row in df.iterrows():
            url = row[url_column]

            # Skip empty URLs
            if pd.isna(url) or not url:
                continue

            logger.info(f"Processing URL {index+1}/{len(df)}: {url}")

            try:
                result = run_from_url(url, download_esg_only)
                results.append(result)

                # Add delay between URLs
                time.sleep(random.uniform(2, 5))

            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")
                results.append({"url": url, "success": False, "message": str(e)})

        return results

    except Exception as e:
        logger.error(f"Error processing Excel file: {str(e)}")
        return [{"success": False, "message": str(e)}]


# ---------------------------------------------------------------------------
# Example usage for testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Example: Search and download ESG reports for a company
    # search_and_download_esg_reports("Novartis")

    # Example: Crawl a specific website
    # run_from_url("https://www.novartis.com/investors/financial-data/annual-reports")

    # Test WebCrawler directly
    test_crawler = WebCrawler("https://www.ubs.com", depth_limit=1, max_pages=10)
    pdf_links = test_crawler.start()
    print(f"Found {len(pdf_links)} PDF links")

    # After finding PDF links
    if pdf_links:
        print(f"Found {len(pdf_links)} PDF links")

        # Create download folder
        download_folder = "downloads/ubs"
        os.makedirs(download_folder, exist_ok=True)

        # Filter for ESG reports (optional)
        esg_pdf_links = filter_esg_reports(pdf_links, "ubs")
        print(f"Filtered to {len(esg_pdf_links)} ESG-related PDF links")

        # Download the PDFs
        downloaded_files = []
        for link in esg_pdf_links:
            try:
                file_path = download_pdf(link['url'], download_folder, "UBS")
                if file_path:
                    downloaded_files.append(file_path)
                    print(f"Downloaded: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Error downloading {link['url']}: {str(e)}")

        print(f"Downloaded {len(downloaded_files)} files to {os.path.abspath(download_folder)}")
