from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.retriever import get_retriever

with open("data/docs.txt") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = splitter.create_documents([text])
retriever = get_retriever(docs)


print("✅ Vector store built")
