from abc import ABC, abstractmethod
from typing import List

from .models import Review


class ReviewClient(ABC):
    """Interface for anything that can supply reviews for an app.

    Everything downstream of collection (validation, storage, processing,
    analysis) should depend on this interface, not on a concrete client.
    That means the data source can later be swapped or supplemented
    (e.g. a scraper, a paid API, a cached DB-backed client) without
    touching any other layer.
    """

    @abstractmethod
    def get_reviews(self, app_id: str, limit: int = 100, country: str = "us") -> List[Review]:
        """Return up to `limit` reviews for the given app_id.

        Raises a subclass of ReviewCollectionError on failure. Never
        returns fabricated or partial-looking-like-complete data silently.
        """
        raise NotImplementedError
