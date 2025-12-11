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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from src.utils.pdf_loader import pdf_loader 

if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError("Please set the GOOGLE_API_KEY in environment variables.")

PDF = pdf_loader()
folder_path = "./data_source"
raw_docs = PDF.load_pdfs_from_folder(folder_path)

if not raw_docs:
    raise RuntimeError("❌ No PDFs found in the folder, or all PDFs failed to extract any text.")
    #print("⚠️ No PDFs found in the folder, The answer will be generated using only the information already available in the knowledge base.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
)
chunks = text_splitter.split_documents(raw_docs)
print(f"Split into {len(chunks)} chunks.")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="demo_rag",
    persist_directory="./chroma_db",
)

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

frontend_path = os.path.join(os.path.dirname(__file__), "../../frontend")
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
        filename=request.filename
    )
    return {"answer": answer}
