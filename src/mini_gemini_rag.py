import os
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from utils.pdf_loader import pdf_loader 
from langchain_core.documents import Document
from models.rag import MyRetriever
from models.chat_session import RagChatSession

if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError("Please set the GOOGLE_API_KEY in your environment variables first.")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

if os.path.exists("../chroma_db"):
    vector_store = Chroma(
        persist_directory="../chroma_db", 
        embedding_function=embeddings,
        collection_name="demo_rag"
    )
else:
    # 這裡可以噴錯，提示使用者先跑 ingest.py
    raise RuntimeError("❌ Vector DB not found! Please run 'python src/ingest.py' first.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # gemini-2.5-flash or gemini-2.5-pro
    temperature=0.5,
)

if __name__ == "__main__":
    print("\n=========================================\n")
    retriever = MyRetriever(vector_store)
    chat = RagChatSession(llm=llm, retriever=retriever)

    while True:
        question = input("What do you wanna ask？(type \"exit\" to exit) > ")
        if question.lower() == "exit":
            break
        answer = chat.ask(
            question,
            k=5,
            score_threshold=1.0,
            doc_type=None,      # 目前只有 pdf，先寫死
            filename=None,
            return_contexts=False # 之後如果要只查某個檔再打開
        )
        print("\nA:", answer)
        print("-" * 60)