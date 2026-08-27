from datetime import date as date_type
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Review(BaseModel):
    review_id: Optional[str] = None
    title: str
    review: str
    rating: int = Field(ge=1, le=5)
    author: Optional[str] = None
    date: Optional[date_type] = None

    @field_validator("title", "review")
    @classmethod
    def strip_whitespace(cls, value: str) -> str:
        return value.strip()