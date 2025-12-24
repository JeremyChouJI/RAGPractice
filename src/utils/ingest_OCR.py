import os

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def ingest_data_unstructured():
    print("🚀 Starting ingestion with Unstructured (OCR Mode)...")
    
    folder_path = "./data_source"
    
    if not os.path.exists(folder_path):
        print("❌ Data directory not found.")
        return

    print(f"📂 Loading PDFs from {folder_path}...")
    print("⏳ This process will be SLOW because it's doing OCR layout analysis.")

    # --- 核心設定 ---
    loader = DirectoryLoader(
        path=folder_path,
        glob="*.pdf",
        loader_cls=UnstructuredPDFLoader,
        loader_kwargs={
            "mode": "elements",           # 解析為獨立元素
            "strategy": "hi_res",         # 啟用高解析度 OCR
            "languages": ["eng"]
        }
    )
    
    try:
        raw_docs = loader.load()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Hint: If the error mentions 'tesseract', check your PATH environment variable.")
        return
    # ----------------
    
    if not raw_docs:
        print("⚠️ No documents loaded.")
        return

    print(f"📄 Loaded {len(raw_docs)} elements.")
    
    # 預覽一下辨識結果，確認中文是否正常
    if len(raw_docs) > 0:
        preview_text = raw_docs[0].page_content[:100].replace('\n', '')
        print(f"🔍 Preview: {preview_text}...")

    # 切分與儲存 (維持原樣)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_docs)
    print(f"📦 Split into {len(chunks)} chunks.")
    # --- 【核心修改開始】 ---
    print("🧹 Cleaning complex metadata for ChromaDB...")
    # 過濾掉 ChromaDB 不支援的複雜 Metadata (如座標資訊)
    # 這一步會把 dict 或 list 類型的 metadata 刪掉，只留簡單型別
    chunks = filter_complex_metadata(chunks)
    # --- 【核心修改結束】 ---

    if "GOOGLE_API_KEY" not in os.environ:
        print("⚠️ Warning: GOOGLE_API_KEY not set.")
    else:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        if os.path.exists("./chroma_db_eng"):
            print("⚠️  Appending to existing DB...")

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="demo_rag",
            persist_directory="./chroma_db_eng",
        )
        print("✅ Ingestion complete! Data saved.")

if __name__ == "__main__":
    ingest_data_unstructured()