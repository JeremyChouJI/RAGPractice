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
DATASET_PATH = "./evaluation/golden_dataset/drcd_golden.json"
TEST_SAMPLE_SIZE = 5 

def main():
    if "GOOGLE_API_KEY" not in os.environ:
        raise RuntimeError("❌ Please set the GOOGLE_API_KEY in your environment variables first.")

    print(f"🔄 Loading existing vector database: {DB_PATH}")
    
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    task_type="retrieval_query"
    )

    if not os.path.exists(DB_PATH):
        raise RuntimeError(f"❌ Vector DB_PATH:{DB_PATH} not found! Please run 'python ingest_txt.py' first.")

    vector_store = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embeddings,
        collection_name="demo_rag"
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", #gemini-2.5-flash-lite
        temperature=0, 
    )

    retriever = MyRetriever(vector_store)
    chat = RagChatSession(llm=llm, retriever=retriever)

    print(f"📖 Loading test questions: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
         raise RuntimeError(f"❌ Dataset not found: {DATASET_PATH}. Did you run 'prepare_full_rag_data.py'?")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    total_questions = len(test_data)
    if TEST_SAMPLE_SIZE and total_questions > TEST_SAMPLE_SIZE:
        print(f"⚠️ Sampling Enabled: Running only first {TEST_SAMPLE_SIZE} questions (out of {total_questions}).")
        target_data = test_data[:TEST_SAMPLE_SIZE]
    else:
        print(f"🚀 Running full benchmark on all {total_questions} questions.")
        target_data = test_data

    results = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    print(f"🚀 Starting benchmark on {len(target_data)} questions...")
    
    for i, item in enumerate(target_data):
        q = item["question"]
        gt = item["ground_truth"]
        
        print(f"[{i+1}/{len(target_data)}] Asking: {q[:30]}...")

        try:
            answer, contexts = chat.ask(
                q, 
                doc_type=None,
                filename=None,
                return_contexts=True,
                score_threshold=0.8
            )
            
            ans_text = answer.content if hasattr(answer, 'content') else str(answer)
            
            results["question"].append(q)
            results["answer"].append(ans_text)
            results["contexts"].append(contexts) 
            results["ground_truth"].append(gt)
            time.sleep(2) 

        except ValueError as ve:
            print(f"❌ Format error: {ve}")
            continue
        except Exception as e:
            print(f"⚠️ Error in question {i+1}: {e}")
            results["question"].append(q)
            results["answer"].append("Error")
            results["contexts"].append([])
            results["ground_truth"].append(gt)
            time.sleep(20) # 出錯時多休息一下

    # 確保輸出資料夾存在
    output_dir = "./evaluation/benchmark_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = f"{output_dir}/rag_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "="*40)
    print(f"✅ Evaluation completed! Processed {len(results['question'])} questions.")
    print(f"📂 Results saved to: {output_file}")
    print("👉 Next step: Run 'python eval.py' to get the scores.")
    print("="*40)

if __name__ == "__main__":
    main()