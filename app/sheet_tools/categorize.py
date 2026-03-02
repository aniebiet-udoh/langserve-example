"""
This is a simple prompt chain that sends a csv file to an LLM for categorization
of the rows based on user-defined categories. The column(s) to use for categorization
are also specified by the user.

Functions:
- categorize_csv(csv_path, categories, columns=None, llm_type='openrouter', output_path=None, max_rows=1000)

The function reads a CSV file, sends the (selected) rows and columns to the configured LLM,
and expects the LLM to return a CSV with the original columns plus a `category` column.
The response is cleaned and returned as a list of dictionaries (one per row). Optionally
saves the CSV output to `output_path` if provided.
"""

from __future__ import annotations

import csv
import io
from typing import List, Optional, Union, Dict

from app.llm.llm import get_llm
from app.pdfconv.utils import CsvProcessor, FileManager


def _rows_to_csv(headers: List[str], rows: List[Dict[str, str]]) -> str:
    """Serialize a list of row dicts to CSV string using given headers."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(headers)
    for r in rows:
        writer.writerow([r.get(h, "") for h in headers])
    return out.getvalue()


def categorize_csv(
    csv_path: str,
    categories: List[str],
    columns: Optional[List[Union[str, int]]] = None,
    hints: Optional[str] = None,
    llm_type: str = "openrouter",
    output_path: Optional[str] = None,
    max_rows: int = 1000,
) -> List[Dict[str, str]]:
    """Categorize rows of a CSV using an LLM.

    Args:
        csv_path: Path to the input CSV file.
        categories: List of category names to use for classification.
        columns: Columns to use for categorization. Can be a list of header names or
                 integer indices. If omitted, all columns are used.
        hints: Optional additional categorization hints to help the LLM do better.
        llm_type: LLM provider to use (passes through to `get_llm`).
        output_path: Optional path to save the LLM-returned CSV.
        max_rows: Maximum number of rows to include in prompt (to prevent huge prompts).

    Returns:
        A list of dictionaries representing the categorized rows. Each dict includes
        the original columns and an additional `category` key.
    """
    # Read CSV
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        lower_headers = [h.strip().lower() for h in (reader.fieldnames or [])]
        headers = reader.fieldnames or []
        rows = list(reader)

    if not headers:
        raise ValueError("Input CSV must have a header row")

    if not rows:
        return []

    # Resolve columns selection
    if columns is None:
        use_cols = headers
    else:
        use_cols = []
        for c in columns:
            if isinstance(c, int):
                try:
                    use_cols.append(headers[c])
                except IndexError:
                    raise ValueError(f"Column index out of range: {c}")
            else:
                if c.lower() not in lower_headers:
                    raise ValueError(f"Column name not found in CSV: {c}")
                use_cols.append(headers[lower_headers.index(c.lower())])

    try:
        from langchain_core.messages import HumanMessage
    except Exception as e:
        raise RuntimeError(
            "langchain-core is required for categorize_csv. Install with: pip install langchain-core"
        ) from e

    llm = get_llm(llm_type)

    out_rows: List[Dict[str, str]] = []
    per_chunk_csv_outputs: List[str] = []

    # Process rows in chunks instead of truncating them. `max_rows` acts as the chunk size.
    total_chunks = (len(rows) + max_rows - 1) // max_rows
    print(f"🔁 Starting categorization: {len(rows)} rows in {total_chunks} chunk(s) (chunk size={max_rows})")

    for chunk_index, start in enumerate(range(0, len(rows), max_rows), start=1):
        chunk_rows = rows[start : start + max_rows]
        print(f"➡️  Processing chunk {chunk_index}/{total_chunks}: rows {start + 1}-{start + len(chunk_rows)}...")

        # Build CSV snippet for this chunk: include only selected columns plus an index column so results can be merged back
        INDEX_COL = "__row_index"
        snippet_headers = use_cols + [INDEX_COL]
        csv_snippet = _rows_to_csv(
            snippet_headers,
            [
                {**{h: r.get(h, "") for h in use_cols}, INDEX_COL: str(start + i)}
                for i, r in enumerate(chunk_rows)
            ],
        )

        # Include chunk context in prompt to help the model (optional but helpful)
        prompt = (
            f"Rows {start + 1}-{start + len(chunk_rows)} of {len(rows)}.\n"
            f"You are given a CSV with columns: {', '.join(snippet_headers)}.\n"
            f"Assign exactly one of these categories to each row: {', '.join(categories)}.\n"
            f"Let your category of choice be based on the content of these columns: {', '.join(use_cols)}.\n"
            f"Return a CSV that contains these columns in this exact order: {','.join(snippet_headers + ['Category'])}.\n"
            "Only return the CSV (no explanation, no markdown, no numbering)."
            f"Ensure that the header row is exactly: {','.join(snippet_headers + ['Category'])}\n\n"
        )

        if hints:
            prompt += f"Additional hints to help categorization:\n\n{hints}\n\n"

        prompt += f"CSV:\n{csv_snippet}\n"

        message = HumanMessage(content=prompt)

        # Call the LLM for this chunk
        print(f"   ⏳ Sending chunk {chunk_index}/{total_chunks} to LLM (provider={llm_type})...")
        response = llm.invoke([message])
        resp_content = getattr(response, "content", str(response))
        print(f"   ✅ Received response for chunk {chunk_index}/{total_chunks}")

        # Clean and parse response
        cleaned_csv = CsvProcessor.clean_response(resp_content)
        parsed = csv.DictReader(io.StringIO(cleaned_csv))

        if not parsed.fieldnames:
            raise ValueError("LLM response missing header row")

        # Find the index and category fields (case-insensitive)
        index_field = None
        category_field = None
        for fn in parsed.fieldnames:
            if fn and fn.strip().lower() == INDEX_COL.lower():
                index_field = fn
            if fn and fn.strip().lower() == "category":
                category_field = fn

        if index_field is None or category_field is None:
            raise ValueError("LLM response missing required 'index' or 'category' column")

        parsed_rows = [dict(r) for r in parsed]

        # Validate row counts match to avoid mixing up data
        if len(parsed_rows) != len(chunk_rows):
            raise ValueError(
                f"LLM returned {len(parsed_rows)} rows for chunk starting at {start}, expected {len(chunk_rows)}"
            )

        # Build a mapping from index -> category
        idx_to_category: Dict[str, str] = {}
        for r in parsed_rows:
            idx = r.get(index_field)
            if idx in idx_to_category:
                raise ValueError(f"Duplicate index {idx} in LLM response")
            idx_to_category[idx] = r.get(category_field, "")

        # Merge categories into the original chunk rows using the index
        for i, original_row in enumerate(chunk_rows):
            idx_str = str(start + i)
            if idx_str not in idx_to_category:
                raise ValueError(f"LLM response missing row with index {idx_str}")
            merged = dict(original_row)
            merged["category"] = idx_to_category[idx_str]
            out_rows.append(merged)

        # Keep original LLM CSV for debugging but don't rely on it for saving final output
        per_chunk_csv_outputs.append(cleaned_csv)

    # Optionally save combined CSV result (original headers + category)
    if output_path and out_rows:
        combined_headers: List[str] = headers + ["category"]
        combined_csv = _rows_to_csv(combined_headers, out_rows)
        FileManager.save_to_file(combined_csv, output_path, "categorization output")
        print(f"✅ Saved combined categorization CSV to {output_path}")

    return out_rows
