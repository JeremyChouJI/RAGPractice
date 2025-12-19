import os
import json
import time
from langchain_community.vectorstores import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from src.models.rag import MyRetriever
from src.models.chat_session import RagChatSession

DB_PATH = "./chroma_db"
DATASET_PATH = "./evaluation/golden_dataset/golden_dataset.json"

def main():
    if "GOOGLE_API_KEY" not in os.environ:
        raise RuntimeError("❌ Please set the GOOGLE_API_KEY in your environment variables first.")

    print(f"🔄 Loading existing vector database: {DB_PATH}")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

    if not os.path.exists(DB_PATH):
        raise RuntimeError(f"❌ Vector DB_PTAH:{DB_PATH} not found! Please run 'python src/utils/ingest.py' first.")

    vector_store = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embeddings,
        collection_name="demo_rag"
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", #gemini-2.5-flash-lite
        temperature=0.5,
    )

    retriever = MyRetriever(vector_store)
    chat = RagChatSession(llm=llm, retriever=retriever)

    print(f"📖 Loading test questions: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        raise RuntimeError(f"❌ DATASET_PATH:{DATASET_PATH} not found! Please run 'python evaluation/create_golden_dataset.py' first.")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 準備結果容器 (符合 RAGAS 格式)
    results = {
        "question": [],
        "answer": [],
        "contexts": [],     # 這必須是 list of list of strings
        "ground_truth": []
    }

    print(f"🚀 Starting automatic evaluation, with a total of {len(test_data)} questions...\n")

    for i, item in enumerate(test_data):
        q = item["question"]
        gt = item["ground_truth"]
        
        print(f"[{i+1}/{len(test_data)}] Asking: {q}")
        
        try:
            # contexts 必須是一個 list，裡面裝著檢索到的文字片段
            answer, contexts = chat.ask(
                q,
                k=5,
                score_threshold=1.0,
                doc_type=None,
                filename=None,
                return_contexts=True
            )
            
            ans_text = answer.content if hasattr(answer, 'content') else str(answer)
            
            results["question"].append(q)
            results["answer"].append(ans_text)
            results["contexts"].append(contexts) 
            results["ground_truth"].append(gt)
            print("💤 Pausing for 20 seconds to avoid rate limiting (cooling down)...")
            time.sleep(20)

        except ValueError as ve:
            print(f"❌ Format error (possibly because chat.ask did not return two values): {ve}")
            break
        except Exception as e:
            print(f"⚠️ An unknown error occurred in question {i+1}: {e}")
            results["question"].append(q)
            results["answer"].append("Error")
            results["contexts"].append([])
            results["ground_truth"].append(gt)
            print("💤 Pausing for 30 seconds to avoid rate limiting (cooling down)...")
            time.sleep(20)

    output_file = "./evaluation/benchmark_output/rag_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "="*40)
    print(f"✅ Evaluation completed! The results have been saved to {output_file}")
    print("👉 Next step: please run python evaluation.py to calculate the score")
    print("="*40)

if __name__ == "__main__":
    main()