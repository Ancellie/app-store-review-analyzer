import logging
from typing import Optional

from playwright.sync_api import (
    Error as PlaywrightError,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

from .exceptions import AppNotFoundError, AppStoreRequestError
from .review_parser import REVIEW_CARD_SELECTOR

logger = logging.getLogger(__name__)

APP_STORE_URL_TEMPLATE = "https://apps.apple.com/{country}/app/{slug}/id{app_id}"
FALLBACK_SLUG = "app"  # Apple's routing only keys off the numeric id; slug text is cosmetic.

NAVIGATION_TIMEOUT_MS = 20_000
SELECTOR_TIMEOUT_MS = 10_000

# We only want the rendered DOM, not images/fonts/media -- faster loads,
# no anti-bot circumvention involved, just not fetching bytes we don't use.
_BLOCKED_RESOURCE_TYPES = {"image", "media", "font"}


class AppleStoreScraper:
    """Renders a public apps.apple.com page with Playwright and returns its HTML.

    This class owns browser lifecycle only: launch, navigate, wait, close.
    It knows nothing about what a Review looks like -- that's
    review_parser's job. Splitting it this way means the parsing logic
    can be unit-tested against static HTML fixtures with no browser
    involved at all, and the browser logic can be swapped (e.g. for a
    remote browser service) without touching the parser.
    """

    def get_rendered_reviews_html(self, app_id: str, country: str = "us", slug: Optional[str] = None) -> str:
        url = APP_STORE_URL_TEMPLATE.format(
            country=country, slug=slug or FALLBACK_SLUG, app_id=app_id
        )

        with sync_playwright() as playwright:
            browser = None
            try:
                browser = playwright.chromium.launch(
                    headless=False,
                    slow_mo=10000,
                )
                page = browser.new_page()
                page.set_default_navigation_timeout(NAVIGATION_TIMEOUT_MS)
                page.set_default_timeout(SELECTOR_TIMEOUT_MS)
                page.route("**/*", self._maybe_block_resource)

                try:
                    response = page.goto(url, wait_until="domcontentloaded")
                except PlaywrightTimeoutError as exc:
                    raise AppStoreRequestError(f"Timed out loading {url}") from exc
                except PlaywrightError as exc:
                    raise AppStoreRequestError(f"Failed to load {url}: {exc}") from exc

                if response is not None and response.status == 404:
                    raise AppNotFoundError(f"App with id={app_id} not found at {url}")

                try:
                    page.wait_for_selector(REVIEW_CARD_SELECTOR, timeout=SELECTOR_TIMEOUT_MS)
                except PlaywrightTimeoutError:
                    # Either the app genuinely has no reviews, or the page
                    # structure has changed since this was written. We
                    # can't distinguish those cases from here, so return
                    # whatever rendered and let the caller (and its tests)
                    # decide -- rather than silently hiding the failure.
                    logger.warning(
                        "No review cards appeared for app_id=%s within %dms",
                        app_id, SELECTOR_TIMEOUT_MS,
                    )

                return page.content()
            finally:
                if browser is not None:
                    browser.close()

    @staticmethod
    def _maybe_block_resource(route) -> None:
        if route.request.resource_type in _BLOCKED_RESOURCE_TYPES:
            route.abort()
        else:
            route.continue_()
