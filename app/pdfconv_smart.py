import base64
import os
from typing import Optional, Generator
import io

class PdfConverterException(Exception):
    """Base exception for PDF converter errors."""
    pass

class PdfConverter():
    def __init__(self):
        # Lazily load environment variables when an instance is created
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            # If dotenv isn't available it's fine; environment may already be configured
            pass

        self.llms = {}
        self.already_converted = []

    def create_client(self, llm_type, max_retries=3):
        """Create LLM client with built-in retry logic.

        Imports of langchain providers are done lazily so module import doesn't fail
        if optional providers are not installed.
        """
        # Common retry configuration (kept for readability)
        retry_config = {
            "max_retries": max_retries,
            "retry_on_exceptions": True,
        }

        if llm_type == "openai":
            try:
                from langchain_openai import ChatOpenAI
            except Exception as e:
                raise PdfConverterException(
                    "ChatOpenAI is required for 'openai' but is not installed or failed to import."
                ) from e

            return ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                max_retries=max_retries,
                timeout=120,
            )

        if llm_type == "openrouter":
            try:
                from langchain_openai import ChatOpenAI
            except Exception as e:
                raise PdfConverterException(
                    "ChatOpenAI (openrouter support) is required but not installed."
                ) from e

            return ChatOpenAI(
                model="nvidia/nemotron-3-nano-30b-a3b:free",
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                openai_api_base="https://openrouter.ai/api/v1",
                max_retries=max_retries,
                timeout=120,
            )

        if llm_type == "groq":
            try:
                from langchain_groq import ChatGroq
            except Exception as e:
                raise PdfConverterException(
                    "ChatGroq is required for 'groq' but is not installed or failed to import."
                ) from e

            return ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                max_retries=max_retries,
                timeout=120,
            )

        if llm_type == "google":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except Exception as e:
                raise PdfConverterException(
                    "ChatGoogleGenerativeAI is required for 'google' but is not installed or failed to import."
                ) from e

            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                google_api_key=os.environ.get("GEMINI_API_KEY"),
                temperature=0,
                max_retries=max_retries,
                timeout=120,
            )

        raise ValueError(f"Unknown llm_type: {llm_type}")

    def get_pdf_page_count(self, pdf_path):
        """Get the number of pages in a PDF."""
        try:
            import PyPDF2  # lazy import
        except Exception as e:
            raise PdfConverterException(
                "PyPDF2 is required to read PDFs but is not installed."
            ) from e

        with open(pdf_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            return len(pdf_reader.pages)

    def split_pdf_into_chunks(self, pdf_path, pages_per_chunk=10):
        """Split a PDF into smaller chunks."""
        try:
            import PyPDF2  # lazy import
        except Exception as e:
            raise PdfConverterException(
                "PyPDF2 is required to chunk PDFs but is not installed."
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

                chunks.append({
                    'data': chunk_bytes.read(),
                    'start_page': start_page + 1,
                    'end_page': end_page,
                    'total_pages': total_pages
                })

        return chunks

    def convert_chunk(self, chunk_data, llm, chunk_info="", is_first_chunk=True,
                     remove_header_if_not_first=False):
        """
        Convert a single PDF chunk to CSV.
        LangChain handles retries automatically.
        """
        try:
            from langchain_core.messages import HumanMessage
        except Exception as e:
            raise PdfConverterException(
                "langchain_core.messages.HumanMessage is required for chunk conversion but is not installed."
            ) from e

        pdf_base64 = base64.b64encode(chunk_data).decode('utf-8')

        prompt = f"Attached is a spreadsheet in PDF{chunk_info}. Convert it to CSV format. "
        prompt += "Only return CSV, do not add any other additional text or response or annotation. Just the csv. "

        if remove_header_if_not_first and not is_first_chunk:
            prompt += "Do not include the header row since this is a continuation of a previous chunk."

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:application/pdf;base64,{pdf_base64}"
                    }
                }
            ]
        )

        # LangChain handles retries, timeouts, and rate limiting automatically
        response = llm.invoke([message])

        if not response or not hasattr(response, 'content'):
            raise ValueError("Invalid response from LLM: no content")

        result = response.content.replace('```csv', '').replace('```', '').strip()

        if not result:
            raise ValueError("LLM returned empty response")

        return result

    def save_partial_results(self, csv_data_list, output_filename, chunk_index):
        """Save partial results when an error occurs."""
        if not output_filename:
            return

        try:
            partial_filename = f"{output_filename}.partial_{chunk_index}"
            with open(partial_filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(csv_data_list))
            print(f"⚠️  Partial results saved to {partial_filename}")
        except Exception as e:
            print(f"❌ Failed to save partial results: {str(e)}")

    def convert(self, input_filename, output_filename=None, llm_type="openrouter",
                max_pages_per_chunk=10, auto_chunk=True, remove_header_if_not_first=False,
                max_retries=3):
        """
        Convert PDF to CSV with automatic chunking for large files.
        """
        # Create client with retry configuration
        if llm_type in self.llms:
            llm = self.llms[llm_type]
        else:
            llm = self.create_client(llm_type, max_retries=max_retries)
            self.llms[llm_type] = llm

        # Check PDF size
        try:
            page_count = self.get_pdf_page_count(input_filename)
            print(f"PDF has {page_count} pages")
        except Exception as e:
            raise PdfConverterException(f"Failed to read PDF: {str(e)}") from e

        # Decide whether to chunk
        should_chunk = auto_chunk and page_count > max_pages_per_chunk

        if should_chunk:
            print(f"Chunking PDF into segments of {max_pages_per_chunk} pages...")
            try:
                chunks = self.split_pdf_into_chunks(input_filename, max_pages_per_chunk)
            except Exception as e:
                raise PdfConverterException(f"Failed to chunk PDF: {str(e)}") from e

            all_csv_data = []
            for i, chunk in enumerate(chunks):
                chunk_info = f" (pages {chunk['start_page']}-{chunk['end_page']} of {chunk['total_pages']})"
                print(f"Converting chunk {i+1}/{len(chunks)}{chunk_info}...")

                try:
                    # LangChain handles all retries automatically
                    csv_data = self.convert_chunk(
                        chunk['data'],
                        llm,
                        chunk_info,
                        is_first_chunk=(i == 0),
                        remove_header_if_not_first=remove_header_if_not_first
                    )

                    # Only remove header if parameter is True and not first chunk
                    if remove_header_if_not_first and i > 0 and csv_data:
                        lines = csv_data.split('\n')
                        if len(lines) > 1:
                            csv_data = '\n'.join(lines[1:])

                    all_csv_data.append(csv_data)

                except Exception as e:
                    # LangChain has already retried, so this is a final failure
                    print(f"❌ Failed to convert chunk {i+1} after all retries: {str(e)}")

                    # Save partial results
                    if all_csv_data:
                        self.save_partial_results(all_csv_data, output_filename, i)
                        result = '\n'.join(all_csv_data)

                        if output_filename:
                            failure_filename = f"{output_filename}.incomplete"
                            try:
                                with open(failure_filename, 'w', encoding='utf-8') as f:
                                    f.write(result)
                                print(f"✓ Saved incomplete results to {failure_filename}")
                            except Exception as save_error:
                                print(f"❌ Failed to save incomplete results: {str(save_error)}")

                        return result
                    else:
                        raise PdfConverterException(
                            f"Failed to convert any chunks. Error on chunk {i+1}: {str(e)}"
                        ) from e

            result = '\n'.join(all_csv_data)
        else:
            # Process entire PDF at once
            print("Converting entire PDF...")
            try:
                with open(input_filename, "rb") as f:
                    pdf_data = f.read()

                # LangChain handles retries automatically
                result = self.convert_chunk(pdf_data, llm, is_first_chunk=True)

            except Exception as e:
                raise PdfConverterException(
                    f"Failed to convert PDF: {str(e)}"
                ) from e

        # Save to file if output filename provided
        if output_filename:
            try:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    f.write(result)
                print(f"✓ Saved to {output_filename}")
            except Exception as e:
                print(f"❌ Failed to save output file: {str(e)}")

        return result

    def convert_streaming(self, input_filename, output_filename=None, llm_type="openrouter",
                         max_pages_per_chunk=10, remove_header_if_not_first=False,
                         max_retries=3) -> Generator[str, None, None]:
        """
        Convert PDF to CSV with streaming responses for each chunk.
        Yields CSV data as it's generated.
        """
        # Create client with retry configuration
        if llm_type in self.llms:
            llm = self.llms[llm_type]
        else:
            llm = self.create_client(llm_type, max_retries=max_retries)
            self.llms[llm_type] = llm

        try:
            page_count = self.get_pdf_page_count(input_filename)
            chunks = self.split_pdf_into_chunks(input_filename, max_pages_per_chunk)
        except Exception as e:
            raise PdfConverterException(f"Failed to prepare PDF: {str(e)}") from e

        output_file = None
        if output_filename:
            try:
                output_file = open(output_filename, 'w', encoding='utf-8')
            except Exception as e:
                print(f"❌ Failed to open output file: {str(e)}")

        all_csv_data = []

        try:
            for i, chunk in enumerate(chunks):
                chunk_info = f" (pages {chunk['start_page']}-{chunk['end_page']} of {chunk['total_pages']})"
                print(f"Converting chunk {i+1}/{len(chunks)}{chunk_info}...")

                try:
                    # LangChain handles retries automatically
                    csv_data = self.convert_chunk(
                        chunk['data'],
                        llm,
                        chunk_info,
                        is_first_chunk=(i == 0),
                        remove_header_if_not_first=remove_header_if_not_first
                    )

                    # Only remove header if parameter is True and not first chunk
                    if remove_header_if_not_first and i > 0:
                        lines = csv_data.split('\n')
                        if len(lines) > 1:
                            csv_data = '\n'.join(lines[1:])

                    all_csv_data.append(csv_data)

                    # Write to file if provided
                    if output_file:
                        try:
                            output_file.write(csv_data)
                            if i < len(chunks) - 1:
                                output_file.write('\n')
                            output_file.flush()
                        except Exception as e:
                            print(f"❌ Failed to write to output file: {str(e)}")

                    # Yield the chunk
                    yield csv_data

                except Exception as e:
                    print(f"❌ Failed to convert chunk {i+1} after all retries: {str(e)}")

                    # Save partial results
                    if all_csv_data and output_filename:
                        self.save_partial_results(all_csv_data, output_filename, i)

                    print(f"💾 Stopping stream at chunk {i+1}.")
                    return

        finally:
            if output_file:
                try:
                    output_file.close()
                except Exception as e:
                    print(f"❌ Failed to close output file: {str(e)}")


# Usage examples: (unchanged)
if __name__ == "__main__":
    converter = PdfConverter()

    # Simple conversion - LangChain handles all retries
    try:
        csv_result = converter.convert(
            "large_statement.pdf",
            "output.csv",
            llm_type="google",
            max_pages_per_chunk=10,
            remove_header_if_not_first=False,
            max_retries=3  # LangChain handles this
        )
        print("✓ Conversion completed successfully!")
    except PdfConverterException as e:
        print(f"Conversion failed: {str(e)}")

    # Streaming conversion
    print("\nStreaming conversion:")
    try:
        chunk_count = 0
        for chunk_csv in converter.convert_streaming(
            "large_statement.pdf",
            "output_streaming.csv",
            llm_type="google",
            max_pages_per_chunk=10,
            remove_header_if_not_first=True,
            max_retries=3
        ):
            chunk_count += 1
            print(f"✓ Received chunk {chunk_count}")
        print(f"✓ Completed! Processed {chunk_count} chunks.")
    except PdfConverterException as e:
        print(f"Streaming failed: {str(e)}")