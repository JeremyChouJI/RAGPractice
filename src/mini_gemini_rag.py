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

PDF = pdf_loader()
folder_path = "./data_source"
raw_docs = PDF.load_pdfs_from_folder(folder_path)

if not raw_docs:
    raise RuntimeError("❌ No PDFs found in the folder, or all PDFs failed to extract any text.")

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
#print("✅   Chroma has been successfully created.")

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
            doc_type= "pdf",      # 目前只有 pdf，先寫死
            # filename="xxx.pdf" # 之後如果要只查某個檔再打開
        )
        print("\nA:", answer)
        print("-" * 60)