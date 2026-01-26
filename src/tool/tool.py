import os
import json
import shutil
import pandas as pd
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2" # 0=INFO, 1=WARNING, 2=ERROR
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("google.generativeai").setLevel(logging.ERROR)
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)
# LangChain Core
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# Google Gemini
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Vector Store & Retrievers
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Re-ranking
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

class HybridRAGToolBuilder:
    def __init__(self, db_path):
        self.db_path = db_path
        self.ensemble_retriever = None
        self.compression_retriever = None
        
        if os.path.exists(db_path):
            try:
                self._initialize_hybrid_retriever()
            except Exception as e:
                print(f"⚠️ RAG initialization failure: {e}")
        else:
            print(f"⚠️ Unable to locate the vector database path: {db_path}. The RAG functionality will be unavailable.")

    def _initialize_hybrid_retriever(self):

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            task_type="retrieval_document"
        )
        
        vector_store = Chroma(
            persist_directory=self.db_path,
            embedding_function=embeddings,
            collection_name="demo_rag"
        )
        
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
        compressor = FlashrankRerank()
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever = self.ensemble_retriever
        )

    def get_tool(self):
        if not self.ensemble_retriever:
            return None

        @tool
        def search_knowledge_base(query: str) -> str:
            """
            Use this tool to search the “articles,” “internal knowledge base,” “technical documentation,” or “regulations.”
            It uses hybrid search, which is especially effective for “proper nouns” or “specific concepts.”
            For the input, please provide the complete query question.
            """
            #docs = self.ensemble_retriever.invoke(query)
            compressed_docs = self.compression_retriever.invoke(query)
            
            result_contexts = []
            for i, doc in enumerate(compressed_docs):
                result_contexts.append(f"--- Document section {i+1} ---\n{doc.page_content}")
            
            return "\n\n".join(result_contexts)

        return search_knowledge_base
