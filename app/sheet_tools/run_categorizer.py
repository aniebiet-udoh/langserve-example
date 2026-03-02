import os
import argparse
from dotenv import load_dotenv

from .categorize import categorize_csv

def run_categorizer(argv=None):
    parser = argparse.ArgumentParser(
        description="Categorize rows in a CSV file using an LLM."
    )
    parser.add_argument("input", help="Input CSV path (use '-' for stdin)")
    parser.add_argument("--output", "-o", help="Output CSV path (use '-' for stdout)", required=True)
    parser.add_argument("--categories", "-c", nargs="+", default=None,
                        help="List of categories to assign to each row")
    parser.add_argument("--categories-file", "-f", default=None,
                        help="Path to a file containing categories to use (one per line)")
    parser.add_argument("--use-cols", "-u", nargs="+", default=None,
                        help="List of column names to use for categorization")
    parser.add_argument("--chunk-size", "-s", type=int, default=100,
                        help="Number of rows to process per LLM request")
    parser.add_argument("--hints", "-H", default=None,
                        help="A markdown file containing additional categorization hints")
    args = parser.parse_args(argv)

    load_dotenv()

    if args.categories is not None and args.categories_file is not None:
        raise ValueError("Cannot specify both --categories and --categories-file")

    categories = args.categories
    if args.categories_file is not None:
        with open(args.categories_file, "r", encoding="utf-8") as f:
            categories = [line.strip() for line in f if line.strip()]

    if args.hints is not None:
        with open(args.hints, "r", encoding="utf-8") as f:
            hints = f.read()
    else:
        hints = None

    categorize_csv(
        csv_path=args.input,
        output_path=args.output,
        categories=categories,
        columns=args.use_cols,
        max_rows=args.chunk_size,
        llm_type=os.getenv("SHEET_CATEGORIZER_LLM_PROVIDER", "openrouter"),
        hints=hints,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(run_categorizer())
