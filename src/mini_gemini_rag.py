import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError("請先在環境變數設定 GOOGLE_API_KEY")

raw_docs = [
    "RAG（Retrieval-Augmented Generation）是一種讓大語言模型在回答問題前，"
    "先到外部知識庫查資料的技術，可以降低幻覺、讓回答更可靠。",

    "在 RAG 系統裡，文件會先被切成很多 chunk，每個 chunk 會被轉成向量，"
    "並存到向量資料庫（例如 Chroma）。查詢時會把問題也轉成向量，拿去做相似度搜尋。",

    "Gemini 是 Google 的生成式模型家族，可以用來做對話、程式生成、"
    "多模態理解，也提供文字 embedding 模型供 RAG 使用。"
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
)

combined_text = "\n\n".join(raw_docs)
chunks = text_splitter.split_text(combined_text)
print(f"切出了 {len(chunks)} 個 chunks")
for i, c in enumerate(chunks, start=1):
    print(f"--- chunk {i} ---\n{c}\n")


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

vector_store = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    collection_name="demo_rag",
    persist_directory="../chroma_db",
)
print("✅Chroma 建立完成")

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # gemini-2.5-flash or gemini-2.5-pro
    temperature=0.2,
)

def rag_answer(question: str) -> str:
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(d.page_content for d in docs)

    #下prompt
    prompt = f"""
             你是一個幫助使用者理解 RAG 與 Gemini 的助理。

             請根據下面的「知識庫內容」回答問題：
             如果資料裡沒有明確提到，請老實說不知道，不要亂編。

             [知識庫內容]
             {context}

             [問題]
             {question}
             """
    #invoke Gemini
    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    print("\n===== 開始測試 RAG 問答 =====\n")
    test_questions = [
        "什麼是 RAG？",
        "為什麼要先把文件切成 chunk？",
        "在這個系統裡向量資料庫扮演什麼角色？",
        "Gemini 在這個 RAG 系統裡可以做什麼？",
    ]

    for q in test_questions:
        print(f"Q: {q}")
        ans = rag_answer(q)
        print("A:", ans)
        print("-" * 60)
