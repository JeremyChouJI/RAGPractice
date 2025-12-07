from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

class MyRetriever:
    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store

    def retriever(
        self,
        query: str,
        *,
        k: int = 5,
        score_threshold: Optional[float] = None,
        doc_type: Optional[str] = None, # "pdf" or "csv" or None
        filename: Optional[str] = None,   # filter with filename
    ) -> List[Tuple[Document, float]]:

        # pass the query into similarity_search_with_score(), Chroma will automatically call your embedding model in the background and embed the query.
        raw_results = self.vector_store.similarity_search_with_score(
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
                if str(doc.metadata.get("source")).lower() != filename.lower():
                    continue

            filtered.append((doc, score))

            if len(filtered) >= k:
                break

        return filtered