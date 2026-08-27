import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

LOOKUP_URL = "https://itunes.apple.com/lookup"
REQUEST_TIMEOUT = (5, 10)  # (connect, read) seconds


def resolve_slug(app_id: str, country: str = "us") -> Optional[str]:
    """Resolve the human-readable URL slug for an app_id via the iTunes Lookup API.

    This hits Apple's free, unauthenticated *metadata* endpoint only --
    it is never used to fetch review content, and it is not the dead
    RSS reviews feed.

    apps.apple.com routing only actually keys off the numeric app_id in
    the URL path; the slug text is cosmetic. So a lookup failure here is
    not fatal -- callers can fall back to a placeholder slug and the page
    still resolves to the correct app.
    """
    try:
        response = requests.get(
            LOOKUP_URL,
            params={"id": app_id, "country": country},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("iTunes lookup failed for app_id=%s (%s): %s", app_id, country, exc)
        return None

    results = payload.get("results") or []
    if not results:
        logger.warning("iTunes lookup returned no results for app_id=%s", app_id)
        return None

    track_view_url = results[0].get("trackViewUrl", "")

    # trackViewUrl looks like: https://apps.apple.com/us/app/spotify.../id324684580
    try:
        return track_view_url.split("/app/")[1].split("/id")[0] or None
    except IndexError:
        logger.warning("Could not parse slug out of trackViewUrl=%r", track_view_url)
        return None
