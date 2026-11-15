"""Command-line interface for the Support Response Generator."""

import argparse
from pathlib import Path

from app.core import (
    NGramModel,
    generate_continuation,
    load_corpus,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="support-response-generator",
        description="Generate customer support response suggestions."
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("hello", help="Check that the application works.")

    suggest_parser = sub.add_parser(
        "suggest",
        help="Generate a continuation for a support response."
    )
    suggest_parser.add_argument(
        "--prompt",
        required=True,
        help="The beginning of the response."
    )
    suggest_parser.add_argument(
        "--length",
        type=int,
        default=20,
        help="Maximum number of generated tokens (default: 20)."
    )
    suggest_parser.add_argument(
        "--n",
        type=int,
        choices=[1, 2, 3],
        default=2,
        help="N-gram size: 1, 2 or 3 (default: 2)."
    )
    suggest_parser.add_argument(
        "--smoothing",
        type=float,
        default=1.0,
        help="Add-k smoothing value (default: 1.0)."
    )

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Support Response Generator is ready.")
        return

    if args.command == "suggest":
        corpus_path = Path("data/support_responses.txt")

        if not corpus_path.exists():
            parser.error(f"Corpus file not found: {corpus_path}")

        if args.length < 1:
            parser.error("--length must be at least 1")

        if args.smoothing <= 0:
            parser.error("--smoothing must be greater than 0")

        corpus = load_corpus(corpus_path)

        if not corpus:
            parser.error("The support corpus is empty.")

        model = NGramModel(n=args.n, k=args.smoothing)
        model.fit(corpus)

        result = generate_continuation(
            model,
            args.prompt,
            max_tokens=args.length
        )

        print("\nSuggested continuation:")
        print(result if result else "No continuation found for this prompt.")


if __name__ == "__main__":
    main()