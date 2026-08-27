import argparse
import json
import logging
import sys

from collector import AppStoreReviewClient, ReviewCollectionError
from collector.fetchlayer_client import FetchLayerReviewClient

# Імпортуємо функцію з нашого сусіднього файлу sentiment.py
from processing.sentiment import attach_sentiment

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
        help="Review provider to use: 'playwright' or 'fetchlayer' (default)",
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

    # 1. Перетворюємо зібрані об'єкти на словники
    review_dicts = [r.model_dump(mode="json") for r in reviews]

    # 2. Викликаємо функцію з файлу sentiment.py для збагачення словників
    enriched_reviews = attach_sentiment(review_dicts, text_field="review")

    # 3. Вивід результату
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(enriched_reviews, f, indent=2, ensure_ascii=False)
        print(f"Saved to {args.out}")
    else:
        for review in enriched_reviews[:10]:
            print("-" * 40)
            print(f"Rating:    {review.get('rating')}")
            print(f"Title:     {review.get('title')}")
            print(f"Author:    {review.get('author')}")
            print(f"Date:      {review.get('date')}")
            print(f"Text:      {review.get('review', '')[:200]}")

            # Виводимо дані тональності
            sentiment_label = review["sentiment"]["label"]
            sentiment_score = review["sentiment"]["compound"]
            print(f"Sentiment: {sentiment_label.upper()} (score: {sentiment_score})")

    return 0


if __name__ == "__main__":
    sys.exit(main())