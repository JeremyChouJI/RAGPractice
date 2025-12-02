import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from loaders.pdf_loader import pdf_loader

if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError("請先在環境變數設定 GOOGLE_API_KEY")

PDF = pdf_loader()
folder_path = "./data_source"
raw_docs = PDF.load_pdfs_from_folder(folder_path)
if not raw_docs.strip():
    raise RuntimeError("❌資料夾內找不到任何 PDF 或 PDF 提取不到文字。")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
)

chunks = text_splitter.split_text(raw_docs)
print(f"切出了 {len(chunks)} 個 chunks")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

vector_store = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    collection_name="demo_rag",
    persist_directory="./chroma_db",
)
print("✅Chroma 建立完成")

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
    return response.content

if __name__ == "__main__":
    print("\n===== PDF Folder RAG 問答測試 =====\n")
    while True:
        question = input("想問什麼？(exit 離開) > ")
        if question.lower() == "exit":
            break
        print("\nA:", rag_answer(question))
        print("-" * 60)