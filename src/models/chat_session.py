# models/chat_session.py
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from ..models.rag import MyRetriever


Role = Literal["user", "assistant"]

@dataclass
class Turn:
    role: Role
    content: str


@dataclass
class RagChatSession:
    llm: ChatGoogleGenerativeAI
    retriever: MyRetriever
    history: List[Turn] = field(default_factory=list)
    max_history_turns: int = 6  # 一次帶進 prompt 的歷史輪數（user+assistant 交錯算一輪）

    def _format_history(self) -> str:
        """把最近幾輪對話整理成文字丟進 prompt。"""

        recent = self.history[-self.max_history_turns:]
        lines = []
        for t in recent:
            prefix = "使用者" if t.role == "user" else "助理"
            lines.append(f"{prefix}: {t.content}")
        return "\n".join(lines)

    def ask(
            self,
            question: str,
            *,
            k: int = 5,
            score_threshold: float = 0.4,
            doc_type: Optional[str] = None,
            filename: Optional[str] = None,
        ) -> str:

        results = self.retriever.retrieve( # MyRetriever，這邊是instance method，instance 是在 mini_gemini_rag.py 建立的
            question,
            k=k,
            score_threshold=score_threshold,
            doc_type=doc_type,
            filename=filename,
        )
        
        docs = [doc for doc, score in results]
        context = "\n\n".join(d.page_content for d in docs)
        history_text = self._format_history()

        prompt = f"""
                 你是一個幫助使用者查詢 PDF / CSV 知識庫的 RAG 助理。
                 請遵守以下規則：
                 - 如果找不到明確答案，可以說不知道，或用你自己的推論，但要標註是推論。
                 - 可以使用先前的對話歷史來理解模糊問題。

                 [對話歷史]
                 {history_text if history_text else "（目前沒有歷史對話）"}

                 [知識庫內容]
                 {context}

                 [使用者最新問題]
                 {question}

                 請用繁體中文回答。
                 """

        response = self.llm.invoke(prompt)
        answer = response.content

        self.history.append(Turn(role="user", content=question))
        self.history.append(Turn(role="assistant", content=answer))

        return answer
