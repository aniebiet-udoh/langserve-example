from langchain.chains import RetrievalQA
from app.llm import get_llm
from app.retriever import get_retriever

def get_rag_chain():
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=get_retriever(),
        chain_type="stuff"
    )
