"""
Re-fetch selected www.acibadem.edu.tr pages after JavaScript (Drupal AJAX) runs.

Some department pages (e.g. academic staff) ship an empty placeholder div
(`acibadem_akademik_kadro_block_v2`); the real list is injected client-side.
`scrape_acibadem` (requests) cannot see that content — this module uses the same
headless Chrome stack as OBS Bologna to capture rendered HTML.

Selenium is imported only inside the functions that need it, so tests can import
`ACIBADEM_JS_URLS` without an optional dependency.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from core.html_extract import extract_title_text_and_embedding_units

if TYPE_CHECKING:
    from selenium.webdriver.remote.webdriver import WebDriver

logger = logging.getLogger(__name__)

# Must match the empty div that gets replaced when faculty data loads.
STAFF_BLOCK_SELECTOR = ".acibadem_akademik_kadro_block_v2"
STAFF_MIN_TEXT_CHARS = 30
BLOCK_WAIT_S = 45
SETTLE_S = 0.7

# English paths that use the client-rendered staff block. Extend as needed.
ACIBADEM_JS_URLS: tuple[str, ...] = (
    "https://www.acibadem.edu.tr/en/academic/undergraduate-programs/faculty-of-engineering-and-natural-sciences/departments/computer-engineering/academic-staff",
)


def _staff_block_has_text(driver: WebDriver) -> bool:
    from selenium.webdriver.common.by import By

    try:
        el = driver.find_element(By.CSS_SELECTOR, STAFF_BLOCK_SELECTOR)
        return len((el.text or "").strip()) >= STAFF_MIN_TEXT_CHARS
    except Exception:
        return False


def wait_for_staff_block(driver: WebDriver, timeout: float = BLOCK_WAIT_S) -> bool:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, STAFF_BLOCK_SELECTOR))
        )
        WebDriverWait(driver, timeout).until(lambda d: _staff_block_has_text(d))
        return True
    except Exception as exc:  # TimeoutException or stale element
        logger.warning("Academic staff block did not become usable: %s", exc)
        return False


def fetch_rendered_page(
    driver: WebDriver, url: str, *, settle: float = SETTLE_S
) -> tuple[str, str, list] | None:
    """
    Navigate to url, wait for the staff list block to populate, return extract triple.
    Returns None if the block never fills (leave existing DB row unchanged).
    """
    driver.set_page_load_timeout(60)
    driver.get(url)
    if not wait_for_staff_block(driver):
        return None
    time.sleep(max(0.0, settle))
    title, text, units = extract_title_text_and_embedding_units(driver.page_source)
    if not text or len(text.strip()) < STAFF_MIN_TEXT_CHARS:
        logger.warning("Rendered page text too short after JS wait: %s", url)
        return None
    return title, text, units


def open_driver() -> WebDriver:
    from core.obs_bologna_scraper import build_driver

    return build_driver()
