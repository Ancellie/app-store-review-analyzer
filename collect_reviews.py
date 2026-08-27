import argparse
import json
import logging
import sys

from collector import AppStoreReviewClient, ReviewCollectionError
from collector.fetchlayer_client import FetchLayerReviewClient

_PROVIDERS = {
    "playwright": AppStoreReviewClient,
    "fetchlayer": FetchLayerReviewClient,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Apple App Store reviews")
    parser.add_argument("app_id", help="Apple App Store numeric app id")
    parser.add_argument("--country", default="us", help="Two-letter storefront code (default: us)")
    parser.add_argument("--limit", type=int, default=100, help="Max reviews to collect (default: 100)")
    parser.add_argument("--out", help="Optional path to write reviews as JSON")
    parser.add_argument(
        "--provider",
        choices=list(_PROVIDERS.keys()),
        default="fetchlayer",
        help="Review provider to use: 'playwright' (Playwright/Apple page) or "
             "'fetchlayer' (FetchLayer API, default)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    client_cls = _PROVIDERS[args.provider]
    try:
        client = client_cls()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    try:
        reviews = client.get_reviews(app_id=args.app_id, limit=args.limit, country=args.country)
    except ReviewCollectionError as exc:
        print(f"Failed to collect reviews: {exc}", file=sys.stderr)
        return 1

    print(f"Collected {len(reviews)} reviews for app_id={args.app_id} (country={args.country})")
    if args.provider == "playwright":
        print("Note: the App Store page exposes only the most-helpful sample, not the full history.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump([r.model_dump(mode="json") for r in reviews], f, indent=2, ensure_ascii=False)
        print(f"Saved to {args.out}")
    else:
        for review in reviews[:3]:
            print("-" * 40)
            print(f"Rating: {review.rating}")
            print(f"Title:  {review.title}")
            print(f"Author: {review.author}")
            print(f"Date:   {review.date}")
            print(f"Text:   {review.text[:200]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())