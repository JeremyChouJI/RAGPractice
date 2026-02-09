import os
import glob

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError("Please set the GOOGLE_API_KEY environment variable.")

def ingest_data():
    print("🚀 Starting ingestion...")
    
    folder_path = "./data_source"

    print("📂 Loading Text files...")
    txt_loader = DirectoryLoader(folder_path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    txt_docs = txt_loader.load()

    # 使用 LangChain 內建的 DirectoryLoader
    #print(f"📂 Loading PDFs from {folder_path}...")
    #loader = PyPDFDirectoryLoader(folder_path)
    raw_docs = txt_docs
    
    print(f"📄 Loaded {len(raw_docs)} document pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        #separators=["\n\n", "\n", " ", ""]
        is_separator_regex=True,
        separators=["\n\n", "(?<=\. )", "(?<=\! )", "(?<=\? )", "(?<=\.\" )", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_docs)
    print(f"📦 Split into {len(chunks)} chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    if os.path.exists("./chroma_db_eng"):
        print("⚠️  Existing DB found. Appending to it (or delete folder to reset).")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="demo_rag",
        persist_directory="./chroma_db_eng",
    )
    print("✅ Ingestion complete! Data saved to ./chroma_db_eng")

if __name__ == "__main__":
    ingest_data()