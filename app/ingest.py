from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("data/docs.txt") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = splitter.create_documents([text])

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

vectorstore.save_local("vectorstore")
print("✅ Vector store built")
