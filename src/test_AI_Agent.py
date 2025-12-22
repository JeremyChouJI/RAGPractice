import os
import pandas as pd
from typing import List

# LangChain Core
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Vector Store & Retrievers
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

if "GOOGLE_API_KEY" not in os.environ:
    print("⚠️ Please set the GOOGLE_API_KEY in your environment variables first.")

# ==========================================
# Tool 1: Hybrid RAG Tool
# ==========================================
class HybridRAGToolBuilder:
    def __init__(self, db_path):
        self.db_path = db_path
        self.ensemble_retriever = None
        
        # 檢查資料庫是否存在
        if os.path.exists(db_path):
            try:
                self._initialize_hybrid_retriever()
            except Exception as e:
                print(f"⚠️ RAG initialization failure: {e}")
        else:
            print(f"⚠️ Unable to locate the vector database path: {db_path}. The RAG functionality will be unavailable.")

    def _initialize_hybrid_retriever(self):
        print(f"Loading Chroma DB: {self.db_path}...")
        
        # 設定 Embedding
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        # 建立 Chroma DB
        vector_store = Chroma(
            persist_directory=self.db_path,
            embedding_function=embeddings,
            collection_name="demo_rag"
        )
        
        # Hybrid Search
        print("Building a Hybrid Search... (Vector + BM25)...")
        existing_data = vector_store.get()
        existing_texts = existing_data['documents']
        existing_metadatas = existing_data['metadatas']

        if not existing_texts:
            print("❌ Chroma DB is empty! Unable to build the RAG tool.")
            return

        # 轉回 Document 物件
        doc_objects = [
            Document(page_content=text, metadata=meta) 
            for text, meta in zip(existing_texts, existing_metadatas)
        ]

        # 建立 BM25 Retriever (關鍵字搜尋)
        bm25_retriever = BM25Retriever.from_documents(documents=doc_objects)
        bm25_retriever.k = 5

        # 建立 Vector Retriever (最原始的語意搜尋)
        chroma_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # 結合兩者成為 Ensemble Retriever (Hybrid Search)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.5, 0.5]
        )
        print("✅ Hybrid RAG is ready")

    def get_tool(self):
        if not self.ensemble_retriever:
            return None

        @tool
        def search_knowledge_base(query: str) -> str:
            """
            使用此工具來搜尋 '內部知識庫'、'技術文件' 或 '規章'。
            它是混合檢索 (Hybrid Search)，對於 '專有名詞' 或 '具體概念' 特別有效。
            輸入請輸入完整的查詢問句。
            """
            # 使用 ensemble retriever 來 retrieve
            docs = self.ensemble_retriever.invoke(query)
            
            # 整理結果回傳給 LLM
            results = []
            for i, doc in enumerate(docs):
                results.append(f"--- Document section {i+1} ---\n{doc.page_content}")
            
            return "\n\n".join(results)

        return search_knowledge_base

# ==========================================
# 工具 2: 數據計算機 (CSV Analysis Tool)
# ==========================================
class SalesDataToolBuilder:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            # 日期欄位轉換
            for col in self.df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        self.df[col] = pd.to_datetime(self.df[col])
                    except:
                        pass
        else:
            print(f"⚠️ Can't find .csv file: {csv_path}")
        
    def get_tool(self):
        if self.df is None:
            return None

        @tool
        def analyze_sales_data(query: str) -> str:
            """
            當問題涉及 '銷售數據'、'統計'、'計算營收' 或 '分析 CSV' 時使用此工具。
            因為是結構化資料，工具會回傳 DataFrame 的統計摘要與範例，LLM 需據此回答。
            """
            info_str = []
            info_str.append(f"Dataset: {self.csv_path}")
            info_str.append(f"Total records: {len(self.df)}")
            info_str.append(f"Fields: {list(self.df.columns)}")
            
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                info_str.append(f"Value statistics:\n{self.df[numeric_cols].describe().to_markdown()}")
            
            info_str.append(f"First 3 rows:\n{self.df.head(3).to_markdown()}")
            
            return "\n".join(info_str)
            
        return analyze_sales_data

def main():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    tools = []
    
    rag_builder = HybridRAGToolBuilder("./chroma_db_eng") 
    rag_tool = rag_builder.get_tool()
    if rag_tool:
        tools.append(rag_tool)

    csv_builder = SalesDataToolBuilder("./data_source/sales_data.csv")
    csv_tool = csv_builder.get_tool()
    if csv_tool:
        tools.append(csv_tool)

    if not tools:
        print("❌ Error: No available tools. Program terminated.")
        return

    # 綁定工具
    llm_with_tools = llm.bind_tools(tools)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an AI assistant. You can use [search_knowledge_base] to query technical documentation,"
            "or use [analyze_sales_data] to analyze sales reports."
            "When encountering a problem, first think about which tool to use, then provide an answer."
        )),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # 建立 Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    print("\n=========================================")
    print("🧠 Agent (Gemini 2.5) + Hybrid RAG Ready")
    print("=========================================\n")
    
    while True:
        q = input("User (輸入 exit 離開) > ")
        if q.lower() in ["exit", "quit"]:
            break
        
        try:
            res = agent_executor.invoke({"input": q})
            print(f"\nAgent: {res['output']}\n")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()