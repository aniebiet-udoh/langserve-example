import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

def get_llm():
    load_dotenv()
    print("API KEY:", os.getenv("OPENAI_API_KEY"))

    return ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID", "nvidia/nemotron-3-nano-30b-a3b:free"),
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    )
