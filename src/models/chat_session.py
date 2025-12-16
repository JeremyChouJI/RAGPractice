from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.models.rag import MyRetriever


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
    max_history_turns: int = 6 

    def ask(
            self,
            question: str,
            *,
            k: int = 5,
            score_threshold: float = 0.4,
            doc_type: Optional[str] = None,
            filename: Optional[str] = None,
            return_contexts: bool = False
        ) -> Union[str, Tuple[str, List[str]]]:

        results = self.retriever.retrieve( # MyRetriever，這邊是instance method，instance 是在 mini_gemini_rag.py 建立的
            question,
            k=k,
            score_threshold=score_threshold,
            doc_type=doc_type,
            filename=filename,
        )
        
        # 整理檢索到的文件
        docs = [doc for doc, score in results]
        context = "\n\n".join(d.page_content for d in docs)
        
        system_prompt = f"""
                 你是一個幫助使用者查詢 PDF [知識庫內容]的 RAG AI 助理。
                 請遵守以下規則：
                 - 如果找不到明確答案，可以說不知道，或用你自己的推論，但要標註是推論。
                 - 可以使用先前的對話歷史來理解模糊問題。

                 [知識庫內容]
                 {context}
                 
                 如果使用者用繁體中文回答，你就用繁體中文回答
                 如果使用者用英文回答，你就用英文回答
                 """
        
        messages = [
            SystemMessage(content=system_prompt), # context 已經在 f-string 裡填進去了
        ]

        for t in self.history[-self.max_history_turns:]:
            if t.role == "user":
                messages.append(HumanMessage(content=t.content))
            else:
                messages.append(AIMessage(content=t.content))

        messages.append(HumanMessage(content=question))

        response = self.llm.invoke(messages)
        answer = response.content

        self.history.append(Turn(role="user", content=question))
        self.history.append(Turn(role="assistant", content=answer))

        if return_contexts:
            # Evaluation mode
            context_texts = [d.page_content for d in docs]
            return answer, context_texts
        else:
            # API mode
            return answer