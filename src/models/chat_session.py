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
            score_threshold: float = 0.8,
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
                You are a RAG AI assistant that helps users query PDF [knowledge base content].
                Please follow the rules below:
                - If a clear answer cannot be found, you may say you don’t know, or provide your own reasoning, but clearly label it as reasoning.
                - You may use previous conversation history to understand ambiguous questions.

                [Knowledge Base Content]
                {context}

                responds in English
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