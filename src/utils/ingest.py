import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pdf_loader import pdf_loader 

if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError("Please set the GOOGLE_API_KEY environment variable.")

def ingest_data():
    print("🚀 Starting ingestion...")
    
    folder_path = "./data_source"
    loader = pdf_loader()
    raw_docs = loader.load_pdfs_from_folder(folder_path)
    
    if not raw_docs:
        print("❌ No documents found.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )
    chunks = text_splitter.split_documents(raw_docs)
    print(f"📦 Split into {len(chunks)} chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="demo_rag",
        persist_directory="./chroma_db",
    )
    print("✅ Ingestion complete! Data saved to ./chroma_db")

if __name__ == "__main__":
    ingest_data()