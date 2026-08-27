from datetime import date as date_type
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Review(BaseModel):
    """A single App Store review, normalized to a stable internal shape.

    This is the contract between the data source (currently a Playwright
    scraper of apps.apple.com) and everything downstream (storage,
    processing, analysis). Downstream code should only ever depend on
    this shape, never on where the data came from.

    Changed from the RSS-based version:
    - `author` was added. The RSS feed didn't reliably expose a reviewer
      name, but the rendered App Store page does (`.author`), and it's
      useful for deduplication and for readable sample reports.
    - `date` is now `date` instead of `datetime`. The RSS feed gave a
      full ISO timestamp; the rendered page only ever shows a calendar
      date (e.g. "11/26/2024" or "Jun 8"), so keeping a time component
      would imply false precision.
    """

    review_id: Optional[str] = None
    title: str
    text: str
    rating: int = Field(ge=1, le=5)
    author: Optional[str] = None
    date: Optional[date_type] = None

    @field_validator("title", "text")
    @classmethod
    def strip_whitespace(cls, value: str) -> str:
        return value.strip()
