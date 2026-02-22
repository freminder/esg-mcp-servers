import os
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Setup logging
logger = logging.getLogger(__name__)


def configure_selenium(download_folder):
    """
    Configure Selenium WebDriver with Chrome using explicitly specified paths.
    Chrome binary and ChromeDriver paths are resolved from the environment variable
    CHROME_BINARY_PATH / CHROMEDRIVER_PATH when set, otherwise the well-known
    installation locations are tried in order.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # Suppress unnecessary errors
    chrome_options.add_argument("--disable-logging")
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

    # Allow overriding via environment variables
    env_chrome_binary = os.environ.get("CHROME_BINARY_PATH", "")
    env_chromedriver = os.environ.get("CHROMEDRIVER_PATH", "")

    # List of possible Chrome binary locations (D drive first, then fallbacks)
    chrome_binary_paths = [
        p for p in [env_chrome_binary] if p
    ] + [
        r"D:\chrome\chrome.exe",
        r"D:\Google\Chrome\Application\chrome.exe",
        r"D:\Program Files\Google\Chrome\Application\chrome.exe",
        r"D:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]

    # List of possible ChromeDriver locations
    chrome_driver_paths = [
        p for p in [env_chromedriver] if p
    ] + [
        r"D:\chromedriver\chromedriver.exe",
        r"D:\chromedriver-win64\chromedriver.exe",
        r"D:\Program Files\chromedriver\chromedriver.exe",
        r"C:\Program Files\chromedriver\chromedriver.exe",
    ]

    # Find Chrome binary
    chrome_binary = None
    for path in chrome_binary_paths:
        if os.path.exists(path):
            chrome_binary = path
            logger.info(f"Found Chrome binary at: {chrome_binary}")
            break

    if not chrome_binary:
        logger.error("Chrome binary not found in any of the expected locations")
        raise FileNotFoundError(
            "Chrome binary not found. Please install Chrome or set the CHROME_BINARY_PATH "
            "environment variable to point to the chrome executable."
        )

    # Find ChromeDriver
    chrome_driver_path = None
    for path in chrome_driver_paths:
        if os.path.exists(path):
            chrome_driver_path = path
            logger.info(f"Found ChromeDriver at: {chrome_driver_path}")
            break

    if not chrome_driver_path:
        logger.error("ChromeDriver not found in any of the expected locations")
        raise FileNotFoundError(
            "ChromeDriver not found. Please download it or set the CHROMEDRIVER_PATH "
            "environment variable to point to the chromedriver executable."
        )

    chrome_options.binary_location = chrome_binary

    # Set download preferences
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": os.path.abspath(download_folder),
        "download.prompt_for_download": False,
        "plugins.always_open_pdf_externally": True,
    })

    # Create service with explicit path
    chrome_service = Service(executable_path=chrome_driver_path)

    # Return Chrome driver with explicit service and options
    try:
        logger.info(f"Creating Chrome WebDriver with binary: {chrome_binary}")
        logger.info(f"Using ChromeDriver at: {chrome_driver_path}")
        driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
        logger.info("Chrome WebDriver initialized successfully")
        return driver
    except Exception as e:
        logger.error(f"Failed to initialize Chrome WebDriver: {e}")
        raise


def accept_cookies(driver, url):
    """Accept cookies on a webpage"""
    driver.get(url)
    try:
        # Wait for the "Accept Cookies" button to be clickable and click it
        accept_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept')]"))
        )
        accept_button.click()
        logger.info("Accepted cookies policy")
    except Exception as e:
        logger.warning(f"Error accepting cookies: {e}")
