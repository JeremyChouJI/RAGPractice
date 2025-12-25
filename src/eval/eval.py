import json
import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

INPUT_FILE = "./src/eval/raw_benchmark.json"
OUTPUT_FILE = "./src/eval/final_score_report.json"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到 {INPUT_FILE}，請先執行 AI_Agent.py 生成回答。")
        return

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

    print(f"⚖️ AI Judge is reviewing {INPUT_FILE} ...")
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    graded_results = []
    
    judge_prompt = ChatPromptTemplate.from_template("""
    You are a fair and strict teacher grading an exam.
    
    [Question]: {question}
    [Student's Answer]: {student_answer}
    [Correct Answer Key]: {ground_truth}
    
    [Task]
    Determine if the Student's Answer conveys the SAME meaning or value as the Correct Answer Key.
    
    [Rules]
    1. If the answer is a number, it must be close (within 5% difference).
    2. If the answer is text, ignore minor wording differences (e.g., "set by" vs "determined by").
    3. If the answer is a list, check if the key items are present.
    4. If the Student's Answer indicates an Error, mark as FAIL.
    
    [Output Format]
    Provide a JSON object with two keys:
    - "reason": A short explanation of why it passes or fails.
    - "pass": boolean true or false.
    
    Output JSON ONLY. No markdown.
    """)

    for i, item in enumerate(records):
        print(f"Processing {i+1}/{len(records)}...", end="\r")
        
        q = item['question']
        resp = item['agent_response']
        gt = str(item['ground_truth'])

        if "ERROR:" in resp:
            item['judge_reason'] = "Agent execution error"
            item['passed'] = False
        else:
            try:
                chain = judge_prompt | llm
                eval_res = chain.invoke({
                    "question": q,
                    "student_answer": resp,
                    "ground_truth": gt
                })
                
                content = eval_res.content.strip()
                if content.startswith("```"):
                    content = content.split("```")[1].replace("json", "").strip()
                
                eval_data = json.loads(content)
                
                item['judge_reason'] = eval_data.get("reason", "No reason provided")
                item['passed'] = eval_data.get("pass", False)
                
            except Exception as e:
                print(f"\n⚠️ Judge Error on item {i}: {e}")
                item['judge_reason'] = f"Judge process failed: {e}"
                item['passed'] = False
        
        graded_results.append(item)

    print("\n\n================ FINAL REPORT ================")
    df = pd.DataFrame(graded_results)
    
    # 最終結果輸出
    final_output = {
        "summary": {
            "total_accuracy": 0.0,
            "total_count": len(graded_results),
            "breakdown": {}
        },
        "details": graded_results
    }
    
    if not df.empty and 'type' in df.columns:
        type_stats_series = df.groupby("type")["passed"].mean().mul(100).round(1)
        
        print(type_stats_series.astype(str) + "%")
        
        total_acc = df['passed'].mean() * 100
        print(f"\n🏆 Total Accuracy: {total_acc:.2f}%")
        
        final_output["summary"]["total_accuracy"] = round(total_acc, 2)
        final_output["summary"]["breakdown"] = type_stats_series.to_dict() 
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
        
    print(f"💾 Graded report saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()