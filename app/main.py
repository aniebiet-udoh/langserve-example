from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from langserve import add_routes

from app.agent import get_agent
# from app.chains import get_rag_chain
import app.custom_router as custom_router

import json

app = FastAPI(
    title="Agentic AI Backend",
    version="1.0"
)

agent = get_agent()
# rag_chain = get_rag_chain()

# add_routes(app, agent, path="/agent")
# add_routes(app, rag_chain, path="/rag")

app.include_router(custom_router.router)

@app.get("/")
def root():
    return {"status": "AI backend running"}
