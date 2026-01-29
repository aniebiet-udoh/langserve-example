from dataclasses import dataclass
from ..llm.config import LLMProviderConfig


@dataclass
class ConversionConfig:
    """Configuration for PDF conversion."""
    max_pages_per_chunk: int = 10
    auto_chunk: bool = True
    remove_header_if_not_first: bool = False
    max_retries: int = 3
    use_structured_messages: bool = True
    extract_text: bool = False

    @classmethod
    def get_max_chunk_pages(cls, llm_type: str, default: int = 10) -> int:
        """Get the maximum pages per chunk for the given LLM provider."""
        if llm_type not in LLMProviderConfig.MODEL_CONFIGS:
            return default
        context_size = LLMProviderConfig.MODEL_CONFIGS.get(llm_type).get("context_size", 100_000)
        return context_size // 10_000 if context_size > 10_000 else 1
