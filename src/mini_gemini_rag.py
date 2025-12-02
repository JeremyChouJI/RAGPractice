import os
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from loaders.pdf_loader import pdf_loader 
from langchain_core.documents import Document

if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError("Please set the GOOGLE_API_KEY in your environment variables first.")

PDF = pdf_loader()
folder_path = "./data_source"
raw_docs = PDF.load_pdfs_from_folder(folder_path)

if not raw_docs:
    raise RuntimeError("❌ No PDFs found in the folder, or all PDFs failed to extract any text.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
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

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # gemini-2.5-flash or gemini-2.5-pro
    temperature=0.5,
)

def rag_answer(question: str) -> str:
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    #下prompt
    prompt = f"""
             以下是從 PDF 資料夾建立的知識庫內容。
             請根據這些內容回答問題， 如果你發現沒有提到的內容，請思考過後給我屬於你的答案

             [知識庫內容]
             {context}

             [問題]
             {question}
            """
    #invoke Gemini
    response = llm.invoke(prompt)

    source_lines = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        file_type = meta.get("type", "unknown")
        page = meta.get("page")

        pos_info = ""
        if file_type == "pdf" and page is not None:
            pos_info = f"Page:{page} "
        else:
            pos_info = "Unknow"

        source_lines.append(f"[{i}] File：{src} | {pos_info}")

    sources_text = "\n".join(source_lines)

    return f"{response.content}\n\n---\nReference：\n{sources_text}"

if __name__ == "__main__":
    print("\n=========================================\n")
    while True:
        question = input("What do you wanna ask？(type \"exit\" to exit) > ")
        if question.lower() == "exit":
            break
        print("\nA:", rag_answer(question))
        print("-" * 60)