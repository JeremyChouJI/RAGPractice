import os
import ast
import pandas as pd
import asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

try:
    from src.tool.tool import HybridRAGToolBuilder
except ImportError:
    from tool import HybridRAGToolBuilder

def build_agent_and_retriever():
    """重建的 Agent 與 Retriever，之後想辦法整合至 Tool"""
    
    if "GOOGLE_API_KEY" not in os.environ:
        print("⚠️ 警告: 請先設定 GOOGLE_API_KEY 環境變數")
    
    print("🧠 Initializing Gemini 2.5 ...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    rag_builder = HybridRAGToolBuilder("./chroma_db_eng") 
    rag_tool = rag_builder.get_tool()
    
    if not rag_tool:
        raise ValueError("❌ Error: 無法建立 Retriever Tool，請檢查向量資料庫路徑。")

    tools = [rag_tool]

    # 設定 Prompt (與 AI_Agent.py 保持一致)
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a specialized Harry Potter Scholar assistant. Your primary goal is to provide accurate information about the Wizarding World based ONLY on the provided textual excerpts.\n\n"
            "CORE CAPABILITIES:\n"
            "1. [search_knowledge_base]: Use this tool to retrieve specific plot details, character dialogues, magical theory, and historical events from the Harry Potter manuscripts.\n\n"
            "STRICT OPERATIONAL GUIDELINES:\n"
            "- GROUNDING: Your answers must be derived solely from the retrieved context. If the information is not present in the search results, state that you do not have enough information from the books to answer, even if you personally 'know' the answer from outside sources.\n"
            "- PRECISION: Pay extremely close attention to proper nouns (names, spells, locations). Ensure that details like specific desserts, points awarded, or exact phrasing of charms match the retrieved text exactly.\n"
            "- STYLE: Maintain a helpful and scholarly tone. When citing facts, try to frame them within the context of the story (e.g., 'According to the text, Hermione mentions...').\n"
            "- THINKING PROCESS: Always think step-by-step. First, identify the key entities in the user's question; second, evaluate which retrieved chunks contain the answer; third, synthesize a response that stays faithful to those chunks."
                    )),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False) # 評測時關閉 verbose 以減少雜訊
    
    raw_retriever = rag_builder.compression_retriever 
    
    return agent_executor, raw_retriever

async def main():
    json_file = "./testset/testset_sample.json"
    if not os.path.exists(json_file):
        print(f"❌ 找不到檔案: {json_file}")
        return

    df = pd.read_json(json_file)
    df = df.dropna(subset=['user_input', 'reference'])

    # 取前 5 筆試跑
    #df = df.head(5) 
    
    print(f"📂 載入測試集，共 {len(df)} 筆資料")

    try:
        agent_executor, retriever = build_agent_and_retriever()
    except Exception as e:
        print(f"❌ 初始化失敗: {e}")
        return

    print("🚀 生成回答與檢索內容...")
    
    questions = df['user_input'].tolist()
    ground_truths = df['reference'].tolist()
    
    answers = []
    contexts = []

    for q in questions:
        try:
            response = agent_executor.invoke({"input": q})
            answers.append(response['output'])
            docs = retriever.invoke(q)
            context_list = [doc.page_content for doc in docs]
            contexts.append(context_list)
            
            print(f"✅ Processed: {q[:30]}...")
            
        except Exception as e:
            print(f"⚠️ Error processing query '{q}': {e}")
            answers.append("Error generating response.")
            contexts.append([])

    clean_answers = []
    for a in answers:
        if isinstance(a, list):
            text_content = ""
            for item in a:
                if isinstance(item, dict) and 'text' in item:
                    text_content += item['text']
                else:
                    text_content += str(item)
            clean_answers.append(text_content)
        else:
            clean_answers.append(str(a))

    clean_contexts = []
    for c_list in contexts:
        clean_contexts.append([str(c) for c in c_list])
    # 準備 Ragas 資料集
    # Ragas 需要的欄位: question, answer, contexts, ground_truth
    ragas_data = {
        "question": questions,
        "answer": clean_answers, 
        "contexts": clean_contexts,
        "ground_truth": ground_truths
    }
    #print(ragas_data)
    
    ragas_dataset = Dataset.from_dict(ragas_data)

    # 執行評測
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
    )
    ragas_llm = LangchainLLMWrapper(llm)
    google_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    ragas_emb = LangchainEmbeddingsWrapper(google_embeddings)
    print("📊 開始 Ragas 評測...")
    
    # 評測指標
    metrics = [
        answer_relevancy,  # 回答是否切題
        answer_correctness,
        faithfulness,      # 回答是否忠於檢索內容 (幻覺檢測)
        context_recall,    # 檢索內容是否包含正確答案所需資訊
        #context_precision, #因為使用哈利波特作為測試先暫時忽略不看

    ]

    results = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb
    )

    print("\n================ 評測結果 ================")
    print(results)
    result_df = results.to_pandas()
    result_df.to_csv("ragas_evaluation_results.csv", index=False)
    result_df.to_json("ragas_evaluation_results.json", index=False)
    print(f"\n💾 詳細評測結果已儲存至")

if __name__ == "__main__":
    asyncio.run(main())