from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

from src.models.rag import MyRetriever
from src.models.chat_session import RagChatSession
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import Chroma

if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError("Please set the GOOGLE_API_KEY in environment variables.")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)
if os.path.exists("./chroma_db"):
    vector_store = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings,
        collection_name="demo_rag"
    )
else:
    # 這裡可以噴錯，提示使用者先跑 ingest.py
    raise RuntimeError("❌ Vector DB not found! Please run 'python src/ingest.py' first.")

retriever = MyRetriever(vector_store)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

session = RagChatSession(
    llm=llm,
    retriever=retriever,
    max_history_turns=6
)

class ChatRequest(BaseModel):
    question: str
    k: int | None = None
    score_threshold: float | None = None
    doc_type: str | None = None
    filename: str | None = None

app = FastAPI()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
frontend_path = os.path.join(BASE_DIR, "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/")
def home():
    return FileResponse(os.path.join(frontend_path, "index.html"))
if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError("Please set the GOOGLE_API_KEY in your environment variables first.")

@app.post("/chat")
def chat(request: ChatRequest):
    answer = session.ask(
        request.question,
        k=request.k,
        score_threshold=request.score_threshold,
        doc_type=request.doc_type,
        filename=request.filename,
        return_contexts=False
    )
    return {"answer": answer}
