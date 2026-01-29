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
                if c not in headers:
                    raise ValueError(f"Column name not found in CSV: {c}")
                use_cols.append(c)

    # Limit rows to avoid overly large prompts
    sampled_rows = rows if len(rows) <= max_rows else rows[:max_rows]

    # Build CSV snippet for prompt using only selected columns
    csv_snippet = _rows_to_csv(use_cols, sampled_rows)

    # Build clear prompt asking for only CSV output with an added 'category' column
    prompt = (
        f"You are given a CSV with columns: {', '.join(use_cols)}.\n"
        f"Assign exactly one of these categories to each row: {', '.join(categories)}.\n"
        "Return a CSV that contains all the original columns in the same order, plus an extra column named 'category'"
        " at the end. Only return the CSV (no explanation, no markdown, no numbering).\n\n"
        "CSV:\n"
        f"{csv_snippet}"
    )

    # Build the message for the LLM
    try:
        from langchain_core.messages import HumanMessage
    except Exception as e:
        raise RuntimeError(
            "langchain-core is required for categorize_csv. Install with: pip install langchain-core"
        ) from e

    message = HumanMessage(content=prompt)

    # Call the LLM
    llm = get_llm(llm_type)
    response = llm.invoke([message])
    resp_content = getattr(response, "content", str(response))

    # Clean response (remove code fences, etc.)
    cleaned_csv = CsvProcessor.clean_response(resp_content)

    # Parse the returned CSV
    out_rows: List[Dict[str, str]] = []
    parsed = csv.DictReader(io.StringIO(cleaned_csv))
    for r in parsed:
        out_rows.append(dict(r))

    # Optionally save result
    if output_path:
        FileManager.save_to_file(cleaned_csv, output_path, "categorization output")

    return out_rows
