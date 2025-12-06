# models/snippet_utils.py
import re
from dataclasses import dataclass
from typing import List, Dict, Any
from langchain_core.documents import Document

@dataclass
class Snippet:
    text: str
    source: str
    metadata: Dict[str, Any]

def _build_snippet_for_doc(question: str, doc: Document, window: int = 80) -> str:
    """簡易的NLP，字串搜尋 + 一個範圍 window，以 **(粗體)** 標註關鍵字。"""
    full_text = doc.page_content
    lower_text = full_text.lower()

    # 把問題拆成關鍵字，因為是用空白切割，這對英文很好用，但對中文的效果較為不佳
    words = re.split(r"\s+", question)
    words = [w for w in words if len(w) >= 2]

    hit_pos = None
    hit_word = None
    for w in words:
        pos = lower_text.find(w.lower())
        if pos != -1:
            hit_pos = pos
            hit_word = w
            break

    if hit_pos is None:
        snippet = full_text[: window * 2]
        return snippet.strip()

    start = max(0, hit_pos - window)
    end = min(len(full_text), hit_pos + len(hit_word) + window)
    snippet = full_text[start:end]

    pattern = re.compile(re.escape(hit_word), re.IGNORECASE)
    snippet = pattern.sub(lambda m: f"**{m.group(0)}**", snippet, count=1)

    return snippet.strip()


def build_snippets(question: str, docs: List[Document], max_snippets: int = 3) -> List[Snippet]:
    """把多個 docs 轉成 snippet list，附上來源資訊。"""
    snippets: List[Snippet] = []

    for doc in docs[:max_snippets]:
        snippet_text = _build_snippet_for_doc(question, doc)
        meta = doc.metadata or {}
        source = meta.get("source") or meta.get("filename") or "unknown"

        snippets.append(
            Snippet(
                text=snippet_text,
                source=str(source),
                metadata=meta,
            )
        )

    return snippets

def format_snippets_for_prompt(snippets):
    lines = []
    for i, s in enumerate(snippets, start=1):
        meta = s.metadata
        src = meta.get("source", "unknown")
        page = meta.get("page")

        extra = []
        if page is not None:
            extra.append(f"page={page}")
        if meta.get("type"):
            extra.append(f"type={meta.get('type')}")

        extra_info = f" ({', '.join(extra)})" if extra else ""

        lines.append(
            f"[{i}] Source：{src}{extra_info}\n{s.text}\n"
        )
    return "\n".join(lines)
