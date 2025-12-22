import os
from typing import List

# LangChain Core components
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Vector Store & Embeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Retrievers for Hybrid Search
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError("Please set the GOOGLE_API_KEY in your environment variables first.")

# 設定 Embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

# 設定 Vector Store (Chroma)
CHROMA_PATH = "./chroma_db_eng"
if not os.path.exists(CHROMA_PATH):
    raise RuntimeError(f"❌ Vector DB not found at {CHROMA_PATH}! Please run your ingest script first.")

vector_store = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
    collection_name="demo_rag"
)

# Hybrid Search
print("正在初始化混合檢索系統 (Vector + BM25)...")

existing_data = vector_store.get() 
existing_texts = existing_data['documents']
existing_metadatas = existing_data['metadatas']

if not existing_texts:
    raise RuntimeError("Chroma DB is empty! Cannot initialize BM25.")

# 將取出的文字轉回 Document 物件
doc_objects = [
    Document(page_content=text, metadata=meta) 
    for text, meta in zip(existing_texts, existing_metadatas)
]

# 建立 BM25 Retriever (關鍵字搜尋)
bm25_retriever = BM25Retriever.from_documents(
    documents=doc_objects
)
bm25_retriever.k = 5

# 建立 Vector Retriever (最原始的語意搜尋)
chroma_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# 結合兩者成為 Ensemble Retriever (Hybrid Search)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever],
    weights=[0.5, 0.5] # 通常關鍵字搜尋 (BM25) 在精確名詞上很強，權重可以設為 0.5 或 0.4
)

# 設定 LLM 與 Prompt
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.3, 
)

system_prompt = (
    "You are a professional AI assistant. Please answer the user's questions based on the [context information] provided below."
    "You must respond in English."
    "If you cannot find the answer in the provided context, please state that you do not know and do not fabricate an answer."
    "\n\n"
    "[Context Information]:\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# RAG Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(ensemble_retriever, question_answer_chain)

if __name__ == "__main__":
    print("\n=========================================")
    print("🤖 Gemini RAG System Ready (Hybrid Search)")
    print("=========================================\n")

    while True:
        try:
            query = input("請輸入問題 (輸入 'exit' 離開) > ")
            if query.lower() in ["exit", "quit"]:
                break
            
            if not query.strip():
                continue

            print("\n正在思考中...\n")
            
            # 執行 Chain
            response = rag_chain.invoke({"input": query})

            print(f"A: {response['answer']}")
            print("-" * 60)

        except Exception as e:
            print(f"Error: {e}")