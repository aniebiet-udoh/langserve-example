try:
    import PyPDF2
except Exception:  # pragma: no cover - import-time fallback for environments without PyPDF2
    PyPDF2 = None
from io import BytesIO
import re
from typing import List

def pdf_to_text(content: bytes) -> str:
    reader = PyPDF2.PdfReader(BytesIO(content))
    txt = []
    for p in reader.pages:
        try:
            txt.append(p.extract_text() or "")
        except:
            txt.append("")
    
    return "\n".join(txt)


def _find_common_prefix(strings: List[str], max_len: int = 1000) -> str:
    if not strings:
        return ""
    # Work on a truncated version to avoid huge comparisons
    s0 = strings[0][:max_len]
    prefix_len = len(s0)
    for s in strings[1:]:
        s = s[:max_len]
        i = 0
        while i < prefix_len and i < len(s) and s0[i] == s[i]:
            i += 1
        prefix_len = i
        if prefix_len == 0:
            return ""
    return s0[:prefix_len]


def _normalize_text(text: str, preserve_newlines: bool = False) -> str:
    # Normalize CRLF
    text = text.replace("\r\n", "\n")
    if preserve_newlines:
        # Collapse multiple blank lines and strip
        text = re.sub(r"\n{2,}", "\n", text).strip()
    else:
        # Replace newlines with spaces and collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
    return text


def pdf_to_csv(content: bytes, output_path: str, *, dedupe_header: bool = True,
               preserve_newlines: bool = False) -> None:
    reader = PyPDF2.PdfReader(BytesIO(content))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")

    # Optionally remove a common prefix (header/footer that repeats on every page)
    if dedupe_header and len(pages) > 1:
        prefix = _find_common_prefix(pages, max_len=1000)
        # If the computed prefix runs into varying content like "page1" vs "page2",
        # prefer stripping only whole-line header content up to the last newline.
        if "\n" in prefix:
            last_nl = prefix.rfind("\n")
            prefix = prefix[: last_nl + 1]

        # Only strip meaningful prefixes (avoid stripping tiny tokens)
        if len(prefix.strip()) >= 10:
            pages = [p[len(prefix):] if p.startswith(prefix) else p for p in pages]

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["page", "text"])
        for i, text in enumerate(pages, start=1):
            text = _normalize_text(text, preserve_newlines=preserve_newlines)
            writer.writerow([i, text])
    return 0

