from typing import Optional, List
from .exceptions import PdfConverterException
from dataclasses import dataclass
import io

@dataclass
class PdfChunk:
    """Represents a chunk of a PDF document."""
    data: bytes
    start_page: int
    end_page: int
    total_pages: int

    @property
    def page_range(self) -> str:
        """Returns a formatted page range string."""
        return f"pages {self.start_page}-{self.end_page} of {self.total_pages}"
    

class PdfUtils:
    """Utilities for PDF manipulation."""

    @staticmethod
    def get_page_count(pdf_path: str) -> int:
        """Get the number of pages in a PDF."""
        try:
            import PyPDF2
        except ImportError as e:
            raise PdfConverterException(
                "PyPDF2 is required to read PDFs but is not installed. Install with: pip install PyPDF2"
            ) from e

        with open(pdf_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            return len(pdf_reader.pages)

    @staticmethod
    def split_into_chunks(pdf_path: str, pages_per_chunk: int = 10) -> List[PdfChunk]:
        """Split a PDF into smaller chunks."""
        try:
            import PyPDF2
        except ImportError as e:
            raise PdfConverterException(
                "PyPDF2 is required to chunk PDFs but is not installed. Install with: pip install PyPDF2"
            ) from e

        chunks = []
        with open(pdf_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            total_pages = len(pdf_reader.pages)

            for start_page in range(0, total_pages, pages_per_chunk):
                end_page = min(start_page + pages_per_chunk, total_pages)
                pdf_writer = PyPDF2.PdfWriter()

                for page_num in range(start_page, end_page):
                    pdf_writer.add_page(pdf_reader.pages[page_num])

                chunk_bytes = io.BytesIO()
                pdf_writer.write(chunk_bytes)
                chunk_bytes.seek(0)

                chunks.append(PdfChunk(
                    data=chunk_bytes.read(),
                    start_page=start_page + 1,
                    end_page=end_page,
                    total_pages=total_pages
                ))

        return chunks

    @staticmethod
    def extract_text(pdf_data: bytes) -> Optional[str]:
        """Extract text from PDF bytes."""
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
            pages_text = []
            for page in reader.pages:
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                if text:
                    pages_text.append(text)
            return "\n".join(pages_text).strip() if pages_text else None
        except Exception:
            return None


class CsvProcessor:
    """Utilities for processing CSV output."""

    @staticmethod
    def clean_response(response_content: str) -> str:
        """Clean LLM response to extract pure CSV."""
        if not response_content:
            raise ValueError("LLM returned empty response")

        # Remove markdown code blocks
        result = response_content.replace('```csv', '').replace('```', '').strip()

        if not result:
            raise ValueError("LLM returned empty CSV after cleaning")

        return result

    @staticmethod
    def remove_header(csv_content: str) -> str:
        """Remove the header row from CSV content."""
        lines = csv_content.split('\n')
        if len(lines) > 1:
            return '\n'.join(lines[1:])
        return csv_content


class FileManager:
    """Handles file I/O operations."""

    @staticmethod
    def save_to_file(content: str, filename: str, description: str = "output"):
        """Save content to a file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Saved {description} to {filename}")
        except Exception as e:
            print(f"❌ Failed to save {description} to {filename}: {str(e)}")

    @staticmethod
    def save_partial_results(csv_data_list: List[str], output_filename: Optional[str], chunk_index: int):
        """Save partial results when an error occurs."""
        if not output_filename:
            return

        try:
            partial_filename = f"{output_filename}.partial_{chunk_index}"
            FileManager.save_to_file(
                '\n'.join(csv_data_list),
                partial_filename,
                f"partial results (up to chunk {chunk_index})"
            )
            print(f"⚠️  Partial results saved to {partial_filename}")
        except Exception as e:
            print(f"❌ Failed to save partial results: {str(e)}")
