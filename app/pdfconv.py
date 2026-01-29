import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert PDF to text or CSV (per-page)."
    )
    parser.add_argument("input", help="Input PDF path (use '-' for stdin)")
    parser.add_argument("--mode", "-m", choices=["basic", "smart"], default="basic",
                        help="Conversion mode: basic (rule-based) or smart (AI-powered)")
    parser.add_argument("--format", "-f", choices=["text", "csv"], default="text",
                        help="Output format: text (print) or csv (save to file)")
    parser.add_argument("--output", "-o", help="Output CSV path (required for csv)")
    parser.add_argument("--no-dedupe", dest="dedupe", action="store_false",
                        help="Do not remove a repeated common header from pages")
    parser.add_argument("--preserve-newlines", dest="preserve_newlines", action="store_true",
                        help="Preserve newlines inside CSV fields (will be quoted)")
    args = parser.parse_args(argv)


    if args.mode == "smart":
        from app.pdfconv_smart import PdfConverter
        converter = PdfConverter()
        converter.convert(
            input_filename=args.input,
            output_filename=args.output,
            llm_type="groq",
        )
        return 0

    else:
        from app.pdfconv_basic import pdf_to_text, pdf_to_csv

        # Read input bytes
        if args.input == "-":
            content = sys.stdin.buffer.read()
        else:
            with open(args.input, "rb") as f:
                content = f.read()

        # Text mode: extract and print all text
        if args.format == "text":
            txt = pdf_to_text(content)
            print(txt)
            return 0

        # CSV mode: write per-page rows to the output file
        if args.format == "csv":
            if not args.output:
                parser.error("Output path is required when format is 'csv'")

            pdf_to_csv(content, args.output, dedupe_header=args.dedupe,
                    preserve_newlines=args.preserve_newlines)
            return 0


if __name__ == "__main__":
    sys.exit(main())