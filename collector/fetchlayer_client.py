"""FetchLayer-based App Store review client.

Uses the FetchLayer API (https://api.fetchlayer.dev) to collect reviews.
The API key is read exclusively from the FETCHLAYER_API_KEY environment
variable.
"""

from __future__ import annotations
from datetime import date

import logging
import math
import os
from typing import List

import requests

from .exceptions import ReviewCollectionError, NoReviewsAvailableError
from .models import Review
from .review_client import ReviewClient

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_ENDPOINT = "https://api.fetchlayer.dev/appstore/reviews"
_REVIEWS_PER_PAGE = 20


class FetchLayerReviewClient(ReviewClient):

    def __init__(self, session: requests.Session | None = None) -> None:
        api_key = os.environ.get("FETCHLAYER_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "FETCHLAYER_API_KEY environment variable is not set or empty. "
                "Export the variable before running the collector."
            )
        self._api_key = api_key
        self._session = session or requests.Session()

    # ------------------------------------------------------------------
    # ReviewClient interface
    # ------------------------------------------------------------------

    def get_reviews(
        self,
        app_id: str,
        limit: int = 100,
        country: str = "us",
    ) -> List[Review]:
        if not app_id or not app_id.strip():
            raise ValueError("app_id must be a non-empty string")
        if limit <= 0:
            raise ValueError("limit must be a positive integer")

        pages_needed = math.ceil(limit / _REVIEWS_PER_PAGE)
        logger.info(
            "FetchLayer: requesting %d page(s) × %d reviews for app_id=%s country=%s",
            pages_needed,
            _REVIEWS_PER_PAGE,
            app_id,
            country,
        )

        raw_reviews = self._fetch_pages(app_id, country, pages_needed)

        if not raw_reviews:
            raise NoReviewsAvailableError(
                f"No reviews returned by FetchLayer for app_id={app_id} country={country}"
            )

        reviews = [self._parse_review(r) for r in raw_reviews[:limit]]
        logger.info("FetchLayer: collected %d review(s)", len(reviews))
        return reviews

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_pages(
        self,
        app_id: str,
        country: str,
        pages: int,
    ) -> list[dict]:
        """Request *pages* pages from the FetchLayer API in a single call.

        The API accepts a ``pages`` parameter, so a single HTTP request
        covers all required pages.  If the response contains fewer
        reviews than expected the client stops early automatically
        (the slice in ``get_reviews`` handles the hard *limit* cap).
        """
        payload = {
            "appId": app_id,
            "country": country,
            "pages": pages,
            "reviewsPerPage": _REVIEWS_PER_PAGE,
            "sort": "recent",
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self._session.post(_ENDPOINT, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise ReviewCollectionError(
                f"FetchLayer API returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise ReviewCollectionError(
                f"Network error while contacting FetchLayer API: {exc}"
            ) from exc

        data = response.json()

        # The API may return a top-level list or wrap reviews in a key.
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("reviews", "data", "results"):
                if isinstance(data.get(key), list):
                    return data[key]

        raise ReviewCollectionError(
            f"Unexpected FetchLayer response shape: {str(data)[:200]}"
        )

    @staticmethod
    def _parse_review(raw: dict) -> Review:
        raw_date = raw.get("date")

        parsed_date = None
        if raw_date:
            parsed_date = date.fromisoformat(raw_date[:10])

        return Review(
            review_id=str(raw.get("id", "") or ""),
            title=str(raw.get("title", "") or ""),
            review=str(raw.get("review", "") or ""),
            rating=int(raw.get("rating", 1)),
            author=raw.get("userName") or None,
            date=parsed_date,
        )