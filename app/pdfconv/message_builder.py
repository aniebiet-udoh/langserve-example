from .exceptions import PdfConverterException
import base64
from .utils import PdfUtils
from .config import LLMProviderConfig

class MessageBuilder:
    """Builds LLM messages with proper formatting for different providers."""

    @staticmethod
    def build_conversion_prompt(
        chunk_info: str = "",
        is_first_chunk: bool = True,
        remove_header_if_not_first: bool = False
    ) -> str:
        """Build the conversion prompt."""
        prompt = f"Attached is a spreadsheet in PDF{chunk_info}. Convert it to CSV format. "
        prompt += "Only return CSV, do not add any other additional text or response or annotation. Just the csv. "

        if remove_header_if_not_first and not is_first_chunk:
            prompt += "Do not include the header row since this is a continuation of a previous chunk."

        return prompt

    @staticmethod
    def build_message(
        prompt: str,
        chunk_data: bytes,
        llm_type: str,
        use_structured_messages: bool,
        extract_text: bool
    ):
        """Build a message for the LLM."""
        try:
            from langchain_core.messages import HumanMessage
        except ImportError as e:
            raise PdfConverterException(
                "langchain_core is required but not installed. Install with: pip install langchain-core"
            ) from e

        # Extract text if requested
        extracted_text = None
        if extract_text:
            extracted_text = PdfUtils.extract_text(chunk_data)

        # Encode PDF as base64 if needed
        pdf_base64 = None
        if not extracted_text:
            pdf_base64 = base64.b64encode(chunk_data).decode('utf-8')

        # Build message based on provider capabilities
        if use_structured_messages and LLMProviderConfig.supports_structured_messages(llm_type):
            content = [{"type": "text", "text": prompt}]

            if extracted_text:
                content.append({"type": "text", "text": extracted_text})
            else:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:application/pdf;base64,{pdf_base64}"}
                })

            return HumanMessage(content=content)
        else:
            # Fallback to single string message
            message_text = prompt
            if extracted_text:
                message_text += "\n\nExtracted text:\n" + extracted_text
            else:
                message_text += "\n\nAttached PDF (base64):\n" + f"data:application/pdf;base64,{pdf_base64}"

            return HumanMessage(content=message_text)
