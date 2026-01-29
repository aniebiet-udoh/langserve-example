from typing import Optional, Generator, List
from .exceptions import PdfConverterException
from .config import ConversionConfig, LLMProviderConfig
from .message_builder import MessageBuilder
from .utils import PdfChunk, PdfUtils, CsvProcessor, FileManager


class PdfConverter:
    """Convert PDF documents to CSV format using LLMs."""

    def __init__(self):
        """Initialize the PDF converter."""
        self._load_environment()
        self.llms = {}
        self.already_converted = []

    def _load_environment(self):
        """Load environment variables from .env file if available."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            # dotenv not installed, assume environment is already configured
            pass

    def _get_or_create_client(self, llm_type: str, max_retries: int):
        """Get cached LLM client or create a new one."""
        if llm_type not in self.llms:
            self.llms[llm_type] = LLMProviderConfig.create_client(llm_type, max_retries=max_retries)
        return self.llms[llm_type]

    def _convert_chunk(
        self,
        chunk_data: bytes,
        llm,
        llm_type: str,
        chunk_info: str = "",
        is_first_chunk: bool = True,
        remove_header_if_not_first: bool = False,
        use_structured_messages: bool = False,
        extract_text: bool = False
    ) -> str:
        """Convert a single PDF chunk to CSV."""
        # Build prompt
        prompt = MessageBuilder.build_conversion_prompt(
            chunk_info=chunk_info,
            is_first_chunk=is_first_chunk,
            remove_header_if_not_first=remove_header_if_not_first
        )

        # Build message
        message = MessageBuilder.build_message(
            prompt=prompt,
            chunk_data=chunk_data,
            llm_type=llm_type,
            use_structured_messages=use_structured_messages,
            extract_text=extract_text
        )

        # Invoke LLM (LangChain handles retries automatically)
        response = llm.invoke([message])

        if not response or not hasattr(response, 'content'):
            raise ValueError("Invalid response from LLM: no content")

        # Clean and return
        return CsvProcessor.clean_response(response.content)

    def _process_chunks(
        self,
        chunks: List[PdfChunk],
        llm,
        llm_type: str,
        config: ConversionConfig,
        output_filename: Optional[str] = None
    ) -> str:
        """Process multiple PDF chunks."""
        all_csv_data = []

        for i, chunk in enumerate(chunks):
            chunk_info = f" ({chunk.page_range})"
            print(f"Converting chunk {i+1}/{len(chunks)}{chunk_info}...")

            try:
                csv_data = self._convert_chunk(
                    chunk_data=chunk.data,
                    llm=llm,
                    llm_type=llm_type,
                    chunk_info=chunk_info,
                    is_first_chunk=(i == 0),
                    remove_header_if_not_first=config.remove_header_if_not_first,
                    use_structured_messages=config.use_structured_messages,
                    extract_text=config.extract_text
                )

                # Remove header if needed
                if config.remove_header_if_not_first and i > 0 and csv_data:
                    csv_data = CsvProcessor.remove_header(csv_data)

                all_csv_data.append(csv_data)

            except Exception as e:
                print(f"❌ Failed to convert chunk {i+1} after all retries: {str(e)}")
                return self._handle_chunk_failure(all_csv_data, output_filename, i, e)

        return '\n'.join(all_csv_data)

    def _handle_chunk_failure(
        self,
        all_csv_data: List[str],
        output_filename: Optional[str],
        chunk_index: int,
        error: Exception
    ) -> str:
        """Handle failure during chunk processing."""
        if all_csv_data:
            # Save partial results
            FileManager.save_partial_results(all_csv_data, output_filename, chunk_index)
            result = '\n'.join(all_csv_data)

            if output_filename:
                failure_filename = f"{output_filename}.incomplete"
                FileManager.save_to_file(result, failure_filename, "incomplete results")

            return result
        else:
            raise PdfConverterException(
                f"Failed to convert any chunks. Error on chunk {chunk_index + 1}: {str(error)}"
            ) from error

    def convert(
        self,
        input_filename: str,
        output_filename: Optional[str] = None,
        llm_type: str = "openrouter",
        max_pages_per_chunk: int = 10,
        auto_chunk: bool = True,
        remove_header_if_not_first: bool = False,
        max_retries: int = 3,
        use_structured_messages: bool = False,
        extract_text: bool = False
    ) -> str:
        """
        Convert PDF to CSV with automatic chunking for large files.

        Args:
            input_filename: Path to input PDF
            output_filename: Path to output CSV (optional)
            llm_type: Type of LLM to use ('openai', 'openrouter', 'groq', 'google')
            max_pages_per_chunk: Maximum pages per chunk (default: 10)
            auto_chunk: Automatically chunk if PDF is large
            remove_header_if_not_first: Remove header row from non-first chunks
            max_retries: Maximum number of retries per chunk
            use_structured_messages: Use structured multimodal messages (provider-dependent)
            extract_text: Extract text from PDF instead of sending as image

        Returns:
            str: The converted CSV content

        Raises:
            PdfConverterException: If conversion fails completely
        """
        config = ConversionConfig(
            max_pages_per_chunk=LLMProviderConfig.get_max_chunk_pages(llm_type, max_pages_per_chunk),
            auto_chunk=auto_chunk,
            remove_header_if_not_first=remove_header_if_not_first,
            max_retries=max_retries,
            use_structured_messages=use_structured_messages,
            extract_text=extract_text
        )

        # Get or create LLM client
        llm = self._get_or_create_client(llm_type, max_retries)

        # Check PDF size
        try:
            page_count = PdfUtils.get_page_count(input_filename)
            print(f"PDF has {page_count} pages")
        except Exception as e:
            raise PdfConverterException(f"Failed to read PDF: {str(e)}") from e

        # Process PDF
        try:
            if config.auto_chunk and page_count > config.max_pages_per_chunk:
                # Chunked processing
                print(f"Chunking PDF into segments of {config.max_pages_per_chunk} pages...")
                chunks = PdfUtils.split_into_chunks(input_filename, config.max_pages_per_chunk)
                result = self._process_chunks(chunks, llm, llm_type, config, output_filename)
            else:
                # Single-pass processing
                print("Converting entire PDF...")
                with open(input_filename, "rb") as f:
                    pdf_data = f.read()

                result = self._convert_chunk(
                    chunk_data=pdf_data,
                    llm=llm,
                    llm_type=llm_type,
                    is_first_chunk=True,
                    use_structured_messages=config.use_structured_messages,
                    extract_text=config.extract_text
                )

        except PdfConverterException:
            raise
        except Exception as e:
            raise PdfConverterException(f"Failed to convert PDF: {str(e)}") from e

        # Save to file if requested
        if output_filename:
            FileManager.save_to_file(result, output_filename, "CSV output")

        return result

    def convert_streaming(
        self,
        input_filename: str,
        output_filename: Optional[str] = None,
        llm_type: str = "openrouter",
        max_pages_per_chunk: int = 10,
        remove_header_if_not_first: bool = False,
        max_retries: int = 3,
        use_structured_messages: bool = False,
        extract_text: bool = False
    ) -> Generator[str, None, None]:
        """
        Convert PDF to CSV with streaming responses for each chunk.

        Yields CSV data as it's generated.

        Args:
            input_filename: Path to input PDF
            output_filename: Path to output CSV (optional, written incrementally)
            llm_type: Type of LLM to use
            max_pages_per_chunk: Maximum pages per chunk
            remove_header_if_not_first: Remove header row from non-first chunks
            max_retries: Maximum number of retries per chunk
            use_structured_messages: Use structured multimodal messages
            extract_text: Extract text from PDF instead of sending as image

        Yields:
            str: CSV data for each successfully converted chunk

        Raises:
            PdfConverterException: If critical errors occur
        """
        config = ConversionConfig(
            max_pages_per_chunk=max_pages_per_chunk,
            remove_header_if_not_first=remove_header_if_not_first,
            max_retries=max_retries,
            use_structured_messages=use_structured_messages,
            extract_text=extract_text
        )

        # Get or create LLM client
        llm = self._get_or_create_client(llm_type, max_retries)

        # Prepare chunks
        try:
            page_count = PdfUtils.get_page_count(input_filename)
            print(f"PDF has {page_count} pages")
            chunks = PdfUtils.split_into_chunks(input_filename, config.max_pages_per_chunk)
        except Exception as e:
            raise PdfConverterException(f"Failed to prepare PDF: {str(e)}") from e

        # Open output file if specified
        output_file = None
        if output_filename:
            try:
                output_file = open(output_filename, 'w', encoding='utf-8')
            except Exception as e:
                print(f"❌ Failed to open output file: {str(e)}")

        all_csv_data = []

        try:
            for i, chunk in enumerate(chunks):
                chunk_info = f" ({chunk.page_range})"
                print(f"Converting chunk {i+1}/{len(chunks)}{chunk_info}...")

                try:
                    csv_data = self._convert_chunk(
                        chunk_data=chunk.data,
                        llm=llm,
                        llm_type=llm_type,
                        chunk_info=chunk_info,
                        is_first_chunk=(i == 0),
                        remove_header_if_not_first=config.remove_header_if_not_first,
                        use_structured_messages=config.use_structured_messages,
                        extract_text=config.extract_text
                    )

                    # Remove header if needed
                    if config.remove_header_if_not_first and i > 0:
                        csv_data = CsvProcessor.remove_header(csv_data)

                    all_csv_data.append(csv_data)

                    # Write to file incrementally
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
                        FileManager.save_partial_results(all_csv_data, output_filename, i)

                    print(f"💾 Stopping stream at chunk {i+1}.")
                    return

        finally:
            if output_file:
                try:
                    output_file.close()
                except Exception as e:
                    print(f"❌ Failed to close output file: {str(e)}")


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    converter = PdfConverter()

    # Example 1: Simple conversion
    try:
        csv_result = converter.convert(
            input_filename="large_statement.pdf",
            output_filename="output.csv",
            llm_type="google",
            max_pages_per_chunk=10,
            remove_header_if_not_first=False,
            max_retries=3
        )
        print("✓ Conversion completed successfully!")
    except PdfConverterException as e:
        print(f"Conversion failed: {str(e)}")

    # Example 2: Streaming conversion
    print("\nStreaming conversion:")
    try:
        chunk_count = 0
        for chunk_csv in converter.convert_streaming(
            input_filename="large_statement.pdf",
            output_filename="output_streaming.csv",
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