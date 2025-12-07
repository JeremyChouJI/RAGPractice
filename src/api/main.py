import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from ..utils.pdf_loader import pdf_loader
from ..models.rag import MyRetriever
from ..models.chat_session import RagChatSession
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI(title="RAG Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError("Please set the GOOGLE_API_KEY in your environment variables first.")

PDF = pdf_loader()

# 如果都是在專案根目錄執行 uvicorn，那這裡用相對路徑就可以
folder_path = "data_source"

raw_docs = PDF.load_pdfs_from_folder(folder_path)
if not raw_docs:
    raise RuntimeError("❌ No PDFs found in the folder, or all PDFs failed to extract any text.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
)
chunks = text_splitter.split_documents(raw_docs)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="demo_rag",
)

my_retriever = MyRetriever(vector_store)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.2,
)

chat_session = RagChatSession(
    llm=llm,
    retriever=my_retriever,
)


# Request / Response schema
class ChatRequest(BaseModel):
    question: str
    k: int = 5
    score_threshold: Optional[float] = 0.4
    doc_type: Optional[str] = None
    filename: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str


# 對外的 API endpoint

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    對話式 RAG 查詢：
    - 會帶上過去幾輪歷史（RagChatSession._format_history）
    - 會用你的 MyRetriever 做檢索與過濾
    """
    answer = chat_session.ask(
        question=req.question,
        k=req.k,
        score_threshold=req.score_threshold,
        doc_type=req.doc_type,
        filename=req.filename,
    )
    return ChatResponse(answer=answer)
    
@app.post("/api/chat", response_model=ChatResponse)
def chat_alias(req: ChatRequest):
    return chat(req)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>RAG Chatbot - Home</title>
    </head>

    <body style="font-family: sans-serif; padding: 40px; line-height: 1.6;">

        <h1>📚 Welcome to the RAG Chatbot API</h1>

        <p>
            這是一個使用 FastAPI + Gemini + Chroma 的 RAG Chatbot API。<br>
            你可以：
        </p>
        <ul>
            <li>上傳 PDF 產生知識庫</li>
            <li>使用 /chat 路由進行問答</li>
            <li>使用 Swagger UI 查看 API 說明文件</li>
        </ul>

        <p>👉 點下面的按鈕進入 API 文件：</p>

        <button onclick="goDocs()" 
                style="padding: 10px 20px; font-size: 16px; cursor: pointer;">
            打開 API 說明文件（/docs）
        </button>

        <script>
        function goDocs() {
            window.location.href = "/docs";
        }
        </script>

        <hr style="margin: 40px 0;">

        <h2>💬 Quick Demo Chat</h2>
        <textarea id="input" rows="3" cols="60"
                  placeholder="輸入你的問題"></textarea><br><br>
        <button onclick="sendMessage()" style="padding: 8px 14px;">Send</button>

        <pre id="output"></pre>

        <script>
        window.onload = function () {
            document.getElementById("input").addEventListener("keydown", function(e) {
                if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
        };
        async function sendMessage() {
            const question = document.getElementById("input").value;
            const res = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question: question,
                    k: 5,
                    score_threshold: null,
                    doc_type: null,
                    filename: null
                })
            });

            const data = await res.json();
            document.getElementById("output").textContent =
                "\\nAssistant: " + data.answer;
        }
        </script>

    </body>
    </html>
    """