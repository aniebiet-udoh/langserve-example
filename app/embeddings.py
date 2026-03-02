

def get_openai_embeddings():
    from langchain.embeddings import OpenAIEmbeddings
    return OpenAIEmbeddings()


def get_huggingface_embeddings():
    # Use HuggingFace embeddings locally (free, runs on your machine)
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
        encode_kwargs={'normalize_embeddings': True}  # Important for cosine similarity
    )

def get_embeddings(docs, provider: str = "huggingface"):
    if provider == "openai":
        return get_openai_embeddings()
    if provider == "huggingface":
        return get_huggingface_embeddings()

    raise ValueError(f"Unknown provider: {provider}")