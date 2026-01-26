import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# 匯入你原本寫好的工具類別
from src.tool.tool import HybridRAGToolBuilder

app = FastAPI()

# --- 1. 初始化 Agent 邏輯 ---
# 這裡將 Agent 設為全域變數，確保服務啟動時只載入一次資料庫
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)
rag_builder = HybridRAGToolBuilder("./chroma_db_eng")
rag_tool = rag_builder.get_tool()

if not rag_tool:
    raise RuntimeError("無法載入 RAG 工具，請檢查資料庫路徑。")

tools = [rag_tool]
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful assistant proficient in analyzing data and retrieving internal knowledge.\n"
        "Currently, You only have access to one powerful tool:\n"
        "1. [search_knowledge_base]: For textual knowledge, technical docs, and policies.\n"
        "Strategy:\n"
        "- If the user asks about 'procedures', 'concepts', or 'textual info', use RAG.\n"
        "- Always think step-by-step."
    )),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# --- 2. 定義資料模型 ---
class ChatRequest(BaseModel):
    question: str

# --- 3. API 路由設定 ---

# 掛載靜態檔案 (假設你的 style.css 和 app.js 在 static 資料夾內)
# 如果檔案就在根目錄，請調整目錄名稱
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    # 回傳首頁
    return FileResponse('static/index.html')

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 呼叫 Agent 進行推論
        response = agent_executor.invoke({"input": request.question})
        return {"answer": response["output"]}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)