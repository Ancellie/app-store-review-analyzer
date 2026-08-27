import argparse
import json
import logging
import sys


from processing.sentiment import attach_sentiment
from processing.transformer_sentiment import attach_sentiment_transformer

def main() -> int:
    parser = argparse.ArgumentParser(description="Test sentiment analysis on local JSON file")
    parser.add_argument("input_file", nargs='?', default="review.json", help="Path to input JSON file (default: review.json)")
    parser.add_argument("--out", help="Optional path to write reviews with sentiment as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            review_dicts = json.load(f)

        if isinstance(review_dicts, dict):
            for key in ['reviews', 'data', 'items', 'results']:
                if key in review_dicts and isinstance(review_dicts[key], list):
                    review_dicts = review_dicts[key]
                    break
            else:
                logging.error(f"JSON має структуру словника з ключами: {list(review_dicts.keys())}. "
                              f"Скрипт не знає, де саме лежать відгуки.")
                return 1

        logging.info(f"Successfully loaded {len(review_dicts)} reviews from {args.input_file}")
    except FileNotFoundError:
        logging.error(f"File not found: {args.input_file}")
        return 1
    except json.JSONDecodeError as exc:
        logging.error(f"Failed to parse JSON in {args.input_file}: {exc}")
        return 1

    logging.info("Applying VADER sentiment analysis...")
    enriched_reviews = attach_sentiment(review_dicts, text_field="review")

    logging.info("Applying Transformer sentiment analysis...")
    enriched_reviews = attach_sentiment_transformer(enriched_reviews, text_field="review", batch_size=16)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(enriched_reviews, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved enriched reviews to {args.out}")
    else:
        for review in enriched_reviews[:10]:
            print("-" * 50)
            print(f"Rating:    {review.get('rating')}")
            print(f"Title:     {review.get('title')}")
            print(f"Text:      {review.get('review', '')[:200]}...")

            vader = review.get("sentiment", {})
            if vader:
                print(f"VADER Sentiment: {vader.get('label', '').upper()} (score: {vader.get('compound')})")

            transformer = review.get("sentiment_transformer", {})
            if transformer:
                t_label = transformer.get('label', '')
                t_score = transformer.get('score', 0.0)
                t_raw = transformer.get('raw_label', '')
                print(f"Transformer:     {t_label.upper()} (confidence: {t_score:.2f} | raw: {t_raw})")

    return 0

if __name__ == "__main__":
    sys.exit(main())