from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

def retrieve_docs(
    vector_store: Chroma,
    query: str,
    *,
    k: int = 5,
    score_threshold: Optional[float] = None,
    doc_type: Optional[str] = None, # "pdf" or "csv" or None
    filename: Optional[str] = None,   # filter with filename
) -> List[Tuple[Document, float]]:

    # pass the query into similarity_search_with_score(), Chroma will automatically call your embedding model in the background and embed the query.
    raw_results = vector_store.similarity_search_with_score(
        query,
        k=max(k + 10, int(k * 3)),
    )

    filtered: List[Tuple[Document, float]] = []

    for doc, score in raw_results:
        if score_threshold is not None:
            if score > score_threshold:
                continue

        if doc_type is not None:
            if str(doc.metadata.get("type")).lower() != doc_type.lower():
                continue

        if filename is not None:
            if str(doc.metadata.get("source")).lower != filename.lower():
                continue

        filtered.append((doc, score))

        if len(filtered) >= k:
            break

    return filtered

def rag_answer(
    llm: ChatGoogleGenerativeAI,
    vector_store: Chroma,
    question: str,
    *,
    k: int = 5,
    score_threshold: float = 0.4,
    doc_type: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    results = retrieve_docs(
        vector_store,
        question,
        k=k,
        score_threshold=score_threshold,
        doc_type=doc_type,
        filename=filename,
    )
    if not results:
        return "我在知識庫裡找不到跟這個問題足夠相關的內容。"
    
    docs = [doc for doc, score in results]
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