import logging
from typing import List, Optional

from .exceptions import NoReviewsAvailableError
from .itunes_lookup import resolve_slug
from .models import Review
from .review_client import ReviewClient
from .review_parser import parse_reviews_html
from .scraper import AppleStoreScraper

logger = logging.getLogger(__name__)


class AppStoreReviewClient(ReviewClient):
    """Collects reviews by rendering the public apps.apple.com page.

    Pipeline: resolve URL slug (iTunes Lookup API, metadata only) ->
    render page (AppleStoreScraper / Playwright) -> parse reviews
    (review_parser, pure function) -> validated Review objects.

    Limitations (see README for full detail):
    - Apple's page exposes roughly the 10 most-helpful reviews, not the
      full review history. This is a sample, not an archive.
    - `limit` is a ceiling, not a guarantee -- if the page shows fewer
      reviews than `limit`, you get fewer reviews back.
    - This does NOT use Apple's RSS reviews feed (dead as of mid-2026),
      any embedded JSON cache (removed from the page), or hashed
      Svelte CSS classes (change on every deploy).
    """

    def __init__(self, scraper: Optional[AppleStoreScraper] = None) -> None:
        self._scraper = scraper or AppleStoreScraper()

    def get_reviews(self, app_id: str, limit: int = 100, country: str = "us") -> List[Review]:
        if not app_id or not app_id.strip():
            raise ValueError("app_id must be a non-empty string")
        if limit <= 0:
            raise ValueError("limit must be a positive integer")

        slug = resolve_slug(app_id, country=country)
        html = self._scraper.get_rendered_reviews_html(app_id=app_id, country=country, slug=slug)
        reviews = parse_reviews_html(html)

        if not reviews:
            raise NoReviewsAvailableError(
                f"No reviews found for app_id={app_id} in country={country}"
            )

        if len(reviews) < limit:
            logger.info(
                "Requested up to %d reviews but the App Store page only exposed %d "
                "for app_id=%s -- this is expected, the page is a sample, not an archive.",
                limit, len(reviews), app_id,
            )

        return reviews[:limit]
