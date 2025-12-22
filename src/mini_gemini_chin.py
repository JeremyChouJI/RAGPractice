import os
import jieba
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError("Please set the GOOGLE_API_KEY in your environment variables first.")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

CHROMA_PATH = "./chroma_db_chin"
if not os.path.exists(CHROMA_PATH):
    raise RuntimeError(f"❌ Vector DB not found at {CHROMA_PATH}! Please run your ingest script first.")

vector_store = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
    collection_name="demo_rag"
)

print("正在初始化混合檢索系統 (Vector + BM25 with Jieba)...")

existing_data = vector_store.get() 
existing_texts = existing_data['documents']
existing_metadatas = existing_data['metadatas']

if not existing_texts:
    raise RuntimeError("Chroma DB is empty! Cannot initialize BM25.")

doc_objects = [
    Document(page_content=text, metadata=meta) 
    for text, meta in zip(existing_texts, existing_metadatas)
]

# 定義中文斷詞函數
def chinese_tokenizer(text: str) -> List[str]:
    """
    使用 jieba 進行中文斷詞。
    BM25 需要 list of tokens，而不是 raw string。
    """
    return jieba.lcut(text)

# 建立 BM25 Retriever (關鍵字搜尋)
# 傳入 preprocess_func=chinese_tokenizer
bm25_retriever = BM25Retriever.from_documents(
    documents=doc_objects,
    preprocess_func=chinese_tokenizer 
)
bm25_retriever.k = 5

chroma_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever],
    weights=[0.5, 0.5] 
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.3, 
)

system_prompt = (
    "你是一個專業的 AI 助手，請根據下方提供的【上下文資訊】來回答使用者的問題。"
    "請務必使用**繁體中文 (Traditional Chinese)** 回答。"
    "如果你在上下文中找不到答案，請直接說你不知道，不要編造答案。"
    "\n\n"
    "【上下文資訊】:\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(ensemble_retriever, question_answer_chain)

if __name__ == "__main__":
    print("\n=========================================")
    print("🤖 Gemini RAG System Ready (Hybrid Search + Jieba)")
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