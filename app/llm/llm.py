from dotenv import load_dotenv

from app.llm.config import LLMProviderConfig

def get_llm(llm_type: str = "openrouter"):

    load_dotenv()

    return LLMProviderConfig.create_client(
        llm_type=llm_type,
        temperature=0,
    )
