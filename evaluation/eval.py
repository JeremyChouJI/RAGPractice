import os
import json
import time
import pandas as pd
from ragas import evaluate
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

BATCH_SIZE = 3      # 每次評測幾筆
SLEEP_SECONDS = 60  # 每批跑完休息幾秒

judge_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_retries=3,
    request_timeout=60
)

eval_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

json_file_path = "./evaluation/benchmark_output/rag_results.json"
print(f"Loading {json_file_path} ...")

try:
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = ["question", "answer", "contexts", "ground_truth"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing key: {key}")

    full_dataset = Dataset.from_dict(data)
    total_samples = len(full_dataset)
    print(f"✅ Data loaded successfully! A total of {total_samples} records. Preparing to process in batches...")

    #分批評測 避免request速度過快
    all_results_df = []

    for i in range(0, total_samples, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total_samples)
        batch_dataset = full_dataset.select(range(i, batch_end))
        
        print(f"\n--- Evaluating records {i+1} to {batch_end} (out of {total_samples} total) ---")
        
        try:
            results = evaluate(
                dataset=batch_dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=judge_llm,
                embeddings=eval_embeddings,
                raise_exceptions=False
            )
            
            batch_df = results.to_pandas()
            all_results_df.append(batch_df)
            print(f"Batch finish！")
            
        except Exception as e:
            print(f"An error occurred during batch execution: {e}")

        if batch_end < total_samples:
            print(f"Pausing for {SLEEP_SECONDS} seconds to avoid rate limiting...")
            time.sleep(SLEEP_SECONDS)

    #合併與輸出
    if all_results_df:
        print("\nMerging all results...")
        final_df = pd.concat(all_results_df, ignore_index=True)
        
        print("\n=== Final Average Score ===")
        numeric_cols = final_df.select_dtypes(include='number').columns
        print(final_df[numeric_cols].mean())

        output_json = "./evaluation/evaluation_result/eval_results.json"
        
        final_df.to_json(output_json, orient='records', force_ascii=False, indent=4)
        
        print(f"\nDetailed evaluation results have been saved to: {output_json}")
    else:
        print("\nNo evaluation results were generated.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")