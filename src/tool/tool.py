import os
import pandas as pd
# LangChain Core
from langchain_core.documents import Document
from langchain_core.tools import tool
# Google Gemini
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Vector Store & Retrievers
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_experimental.agents import create_pandas_dataframe_agent

class HybridRAGToolBuilder:
    def __init__(self, db_path):
        self.db_path = db_path
        self.ensemble_retriever = None
        
        if os.path.exists(db_path):
            try:
                self._initialize_hybrid_retriever()
            except Exception as e:
                print(f"⚠️ RAG initialization failure: {e}")
        else:
            print(f"⚠️ Unable to locate the vector database path: {db_path}. The RAG functionality will be unavailable.")

    def _initialize_hybrid_retriever(self):
        print(f"Loading Chroma DB: {self.db_path}...")
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        vector_store = Chroma(
            persist_directory=self.db_path,
            embedding_function=embeddings,
            collection_name="demo_rag"
        )
        
        print("Building a Hybrid Search... (Vector + BM25)...")
        existing_data = vector_store.get()
        existing_texts = existing_data['documents']
        existing_metadatas = existing_data['metadatas']

        if not existing_texts:
            print("❌ Chroma DB is empty! Unable to build the RAG tool.")
            return

        # BM25Retriever.from_documents(documents -> List[Document])
        doc_objects = [
            Document(page_content=text, metadata=meta) 
            for text, meta in zip(existing_texts, existing_metadatas)
        ]

        bm25_retriever = BM25Retriever.from_documents(documents=doc_objects)
        bm25_retriever.k = 5

        chroma_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

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
            Use this tool to search the “internal knowledge base,” “technical documentation,” or “regulations.”
            It uses hybrid search, which is especially effective for “proper nouns” or “specific concepts.”
            For the input, please provide the complete query question.
            """
            docs = self.ensemble_retriever.invoke(query)
            
            result_contexts = []
            for i, doc in enumerate(docs):
                result_contexts.append(f"--- Document section {i+1} ---\n{doc.page_content}")
            
            return "\n\n".join(result_contexts)

        return search_knowledge_base
    
class SalesDataToolBuilder:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            for col in self.df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        self.df[col] = pd.to_datetime(self.df[col])
                    except:
                        pass
        else:
            print(f"⚠️ Can't find .csv file: {csv_path}")
        
    def get_tool(self, llm):
        """
        這裡需要傳入 llm，因為內部的 Pandas Agent 需要LLM來寫程式
        """
        if self.df is None:
            return None

        pandas_agent = create_pandas_dataframe_agent(
            llm,
            self.df,
            verbose=False,
            allow_dangerous_code_execution=True, #需要設定成True，因為它真的在執行 Python
            handle_parsing_errors=True
        )

        @tool
        def analyze_sales_data(query: str) -> str:
            """
            [Function]: A powerful data analysis tool that can write Python code to perform calculations, statistics, grouping, or plotting.
            [When to use]: Use this when the question involves “data computation,” “category comparison,” “trend analysis,” or “which one sells the best.”
            [Input]: Please pass the user’s original question in full.
            """
            try:
                # 呼叫子代理人 (pandas_agent) 分析、統整資料，最後再交由LLM回答
                response = pandas_agent.invoke({"input": query})
                return response['output']
            except Exception as e:
                return f"An error occurred during the analysis process: {str(e)}"
            
        return analyze_sales_data