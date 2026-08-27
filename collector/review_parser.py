import logging
import re
from datetime import datetime, date as date_type
from typing import List, Optional

from bs4 import BeautifulSoup, Tag

from .models import Review

logger = logging.getLogger(__name__)

# Stable signals only: no svelte-* hashed classes.
# See: https://browserbeam.com/blog/scrape-app-store/ (verified independently, mid-2026)
REVIEW_CARD_SELECTOR = '[aria-labelledby^="review-"]'

_REVIEW_ID_PATTERN = re.compile(r"^review-(\d+)-title$")
_RATING_PATTERN = re.compile(r"(\d+)\s+Stars?", re.IGNORECASE)


def parse_reviews_html(html: str) -> List[Review]:
    """Parse rendered App Store review cards out of a page's HTML.

    Pure function: no browser, no network, no I/O. Takes whatever HTML
    AppleStoreScraper captured (via Playwright) and returns validated
    Review objects. This separation is what makes the parser unit
    testable against static fixtures.
    """
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select(REVIEW_CARD_SELECTOR)

    reviews: List[Review] = []
    seen_keys = set()

    for card in cards:
        if _is_detail_view_clone(card):
            continue

        review = _parse_card(card)
        if review is None:
            continue

        dedup_key = review.review_id or (review.title, review.text, review.rating, review.author)
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        reviews.append(review)

    return reviews


def _is_detail_view_clone(card: Tag) -> bool:
    # The page renders each review twice: a visible summary card and a
    # hidden detail-view clone. Without filtering this, every review
    # would be duplicated (often with some fields empty on the clone).
    classes = card.get("class") or []
    return "is-detail-view" in classes


def _parse_card(card: Tag) -> Optional[Review]:
    title_el = card.find("h3")
    stars_el = card.select_one("ol.stars")
    time_el = card.find("time")
    author_el = card.select_one(".author")
    body_el = card.select_one("p.content")

    # Promo/editorial blocks (e.g. "Editors' Choice") sometimes sit among
    # review cards but carry no time or author. Skip them rather than
    # emit a fake review with missing fields.
    if time_el is None or author_el is None:
        return None

    rating = _parse_rating(stars_el)
    if rating is None:
        logger.debug("Skipping card with no parseable rating: %s", card.get("aria-labelledby"))
        return None

    return Review(
        review_id=_parse_review_id(card.get("aria-labelledby")),
        title=title_el.get_text(strip=True) if title_el else "",
        text=body_el.get_text(strip=True) if body_el else "",
        rating=rating,
        author=author_el.get_text(strip=True),
        date=_parse_date(time_el.get_text(strip=True)),
    )


def _parse_review_id(aria_labelledby: Optional[str]) -> Optional[str]:
    if not aria_labelledby:
        return None
    match = _REVIEW_ID_PATTERN.match(aria_labelledby)
    return match.group(1) if match else None


def _parse_rating(stars_el: Optional[Tag]) -> Optional[int]:
    if stars_el is None:
        return None
    label = stars_el.get("aria-label", "")
    match = _RATING_PATTERN.search(label)
    if not match:
        return None
    rating = int(match.group(1))
    return rating if 1 <= rating <= 5 else None


def _parse_date(raw: str) -> Optional[date_type]:
    """Best-effort date parsing.

    The App Store renders dates inconsistently: older reviews as
    MM/DD/YYYY, recent ones as a relative "Mon D" with no year. Both are
    handled, but a "Mon D" value with no year is assumed to fall in the
    current year -- that's a real limitation of the source, not a
    guarantee, and is documented in the README.
    """
    raw = raw.strip()
    if not raw:
        return None

    try:
        return datetime.strptime(raw, "%m/%d/%Y").date()
    except ValueError:
        pass

    for fmt in ("%b %d", "%B %d"):
        try:
            parsed = datetime.strptime(raw, fmt)
            return parsed.replace(year=datetime.now().year).date()
        except ValueError:
            continue

    logger.debug("Could not parse review date: %r", raw)
    return None
