import os
from dataclasses import dataclass
from .exceptions import PdfConverterException


@dataclass
class ConversionConfig:
    """Configuration for PDF conversion."""
    max_pages_per_chunk: int = 10
    auto_chunk: bool = True
    remove_header_if_not_first: bool = False
    max_retries: int = 3
    use_structured_messages: bool = False
    extract_text: bool = False


# ============================================================================
# LLM Provider Configuration
# ============================================================================

class LLMProviderConfig:
    """Configuration and factory for LLM providers."""

    # Providers that support structured multimodal messages
    STRUCTURED_MESSAGE_PROVIDERS = {"groq"}

    # Default model configurations
    MODEL_CONFIGS = {
        "openai": {
            "model": "gpt-4o",
            "package": "langchain_openai",
            "class": "ChatOpenAI",
        },
        "openrouter": {
            "model": "nvidia/nemotron-3-nano-30b-a3b:free",
            "package": "langchain_openai",
            "class": "ChatOpenAI",
            "api_base": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        },
        "groq": {
            "model": "llama-3.3-70b-versatile",
            "package": "langchain_groq",
            "class": "ChatGroq",
        },
        "google": {
            "model": "gemini-2.0-flash-exp",
            "package": "langchain_google_genai",
            "class": "ChatGoogleGenerativeAI",
            "api_key_env": "GEMINI_API_KEY",
        },
    }

    @classmethod
    def supports_structured_messages(cls, llm_type: str) -> bool:
        """Check if provider supports structured multimodal messages."""
        return llm_type.lower() in cls.STRUCTURED_MESSAGE_PROVIDERS

    @classmethod
    def create_client(cls, llm_type: str, max_retries: int = 3, temperature: float = 0, timeout: int = 120):
        """Create an LLM client for the specified provider."""
        if llm_type not in cls.MODEL_CONFIGS:
            raise ValueError(f"Unknown llm_type: {llm_type}. Available: {list(cls.MODEL_CONFIGS.keys())}")

        config = cls.MODEL_CONFIGS[llm_type]

        # Import the required class
        try:
            module = __import__(config["package"], fromlist=[config["class"]])
            client_class = getattr(module, config["class"])
        except Exception as e:
            raise PdfConverterException(
                f"{config['class']} from {config['package']} is required for '{llm_type}' but failed to import."
            ) from e

        # Build client kwargs
        client_kwargs = {
            "model": config["model"],
            "temperature": temperature,
            "max_retries": max_retries,
            "timeout": timeout,
        }

        # Add provider-specific configuration
        if "api_base" in config:
            client_kwargs["openai_api_base"] = config["api_base"]

        if "api_key_env" in config:
            api_key = os.getenv(config["api_key_env"])
            if llm_type == "google":
                client_kwargs["google_api_key"] = api_key
            else:
                client_kwargs["openai_api_key"] = api_key

        return client_class(**client_kwargs)
