from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

class MyRetriever:
    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        score_threshold: Optional[float] = None,
        doc_type: Optional[str] = None, # "pdf" or "csv" or None
        filename: Optional[str] = None,   # filter with filename
    ) -> List[Tuple[Document, float]]:

        where_filter = {}
        if doc_type:
            where_filter["type"] = doc_type
        if filename:
            where_filter["source"] = filename
        
        if not where_filter:
            where_filter = None

        raw_results = self.vector_store.similarity_search_with_score(
            query,
            k=k, 
            filter=where_filter 
        )

        filtered: List[Tuple[Document, float]] = []
        for doc, score in raw_results:
            if score_threshold is not None:
                 if score > score_threshold:
                    continue
            filtered.append((doc, score))
            
        return filtered