class ReviewCollectionError(Exception):
    """Base exception for all review collection failures."""


class AppNotFoundError(ReviewCollectionError):
    """Raised when the given app_id does not correspond to an app on the store."""


class NoReviewsAvailableError(ReviewCollectionError):
    """Raised when the app exists but no reviews could be retrieved."""


class AppStoreRequestError(ReviewCollectionError):
    """Raised for network failures, timeouts, or persistent rate limiting."""


class AppStoreResponseError(ReviewCollectionError):
    """Raised when the App Store response cannot be parsed as expected."""
