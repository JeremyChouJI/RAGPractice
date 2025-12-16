import json

# 完整的 50 題黃金資料集
# (包含 Attention, GPT-4, Llama, ChatGPT, Gemini 2.5)
data = [
    # --- 1. Attention Is All You Need ---
    {"question": "Transformer 架構完全拋棄了哪兩種傳統的神經網路結構？", "ground_truth": "拋棄了遞歸神經網絡 (RNN) 和卷積神經網絡 (CNN)。"},
    {"question": "Transformer 模型中，「Self-Attention」層的時間複雜度是多少？", "ground_truth": "每層的時間複雜度為 O(n^2 · d)，其中 n 是序列長度，d 是維度。"},
    {"question": "論文中提到的 Multi-Head Attention 有什麼主要優點？", "ground_truth": "它允許模型同時關注來自不同位置的不同表徵子空間 (Representation Subspaces) 的信息。"},
    {"question": "Transformer 在 WMT 2014 英語-德語翻譯任務上達到了多少 BLEU 分數？", "ground_truth": "達到了 28.4 BLEU。"},
    {"question": "為了讓模型知道詞彙在序列中的順序，Transformer 引入了什麼機制？", "ground_truth": "位置編碼 (Positional Encoding)。"},
    {"question": "位置編碼使用了哪兩種函數來計算？", "ground_truth": "使用了正弦 (Sine) 和餘弦 (Cosine) 函數。"},
    {"question": "Transformer 的 Decoder 與 Encoder 在 Attention 機制上有什麼不同？", "ground_truth": "Decoder 的 Self-Attention 層使用了 Masking (遮罩)，確保位置 i 只能關注到 i 之前的位置。"},
    {"question": "論文中使用的 Optimizer 是哪一個？", "ground_truth": "Adam Optimizer。"},
    {"question": "Transformer Base 模型的參數量大約是多少？", "ground_truth": "約 65 Million (65M)。"},
    {"question": "為什麼 Self-Attention 在處理長距離依賴 (Long-range dependencies) 時比 RNN 有優勢？", "ground_truth": "因為 Self-Attention 中任意兩個位置之間的路徑長度都是 O(1)，而 RNN 是 O(n)。"},

    # --- 2. GPT-4 Technical Report ---
    {"question": "根據報告，GPT-4 在美國律師資格考試 (Uniform Bar Exam) 中的表現位於前百分之幾？", "ground_truth": "前 10% (Top 10%)。"},
    {"question": "GPT-4 與 GPT-3.5 相比，主要的輸入模態 (Input Modality) 差異是什麼？", "ground_truth": "GPT-4 能夠接受圖像 (Image) 和文本輸入，而 GPT-3.5 只能接受文本。"},
    {"question": "OpenAI 使用了什麼方法來預測 GPT-4 的訓練損失 (Training Loss)？", "ground_truth": "使用了「可預測的擴展法則 (Predictable Scaling Laws)」，透過較小的模型來推算大模型的表現。"},
    {"question": "GPT-4 在處理「幻覺 (Hallucination)」問題上，比 GPT-3.5 提升了多少百分比的準確度？", "ground_truth": "在內部對抗性真實性評估中，得分比 GPT-3.5 高出 40%。"},
    {"question": "為了安全性，GPT-4 訓練過程中引入了什麼類型的專家紅隊測試 (Red Teaming)？", "ground_truth": "涉及 50 多位來自長期 AI 對齊風險、網絡安全、生物風險等領域的專家。"},
    {"question": "GPT-4 的訓練過程主要分為哪兩個階段？", "ground_truth": "1. 預訓練 (Pre-training) 2. 基於人類反饋的強化學習 (RLHF) 微調。"},
    {"question": "報告中提到 GPT-4 在視覺輸入方面的功能（例如解釋梗圖）是否已對所有使用者開放？", "ground_truth": "報告發布時，視覺輸入功能僅是預覽 (Preview)，尚未對公眾全面開放。"},
    {"question": "RBR (Rule-Based Reward) 模型在 GPT-4 的安全訓練中扮演什麼角色？", "ground_truth": "用於訓練 PPO 模型拒絕有害請求（如製造危險化學品）。"},
    {"question": "GPT-4 的 Context Window 最大版本可以支援多少 tokens？", "ground_truth": "32,768 tokens (約 32k)。"},
    {"question": "OpenAI 是否在技術報告中公開了 GPT-4 的具體參數數量？", "ground_truth": "沒有，報告明確指出不公開模型大小、架構或硬體細節。"},

    # --- 3. Llama & PEFT ---
    {"question": "Llama 1 模型主要是在什麼數據集上訓練的？", "ground_truth": "主要在公開可用的數據集 (Publicly available datasets) 上訓練，沒有使用專有數據。"},
    {"question": "PEFT 的全名是什麼？它的核心目的是什麼？", "ground_truth": "Parameter-Efficient Fine-Tuning。目的是在微調大型語言模型時，只更新少量參數以降低計算成本。"},
    {"question": "LoRA (Low-Rank Adaptation) 的主要技術原理是什麼？", "ground_truth": "凍結預訓練權重，並在 Transformer 層中注入可訓練的低秩分解矩陣 (Rank-decomposition matrices)。"},
    {"question": "Llama 2 相比 Llama 1，在 Context Length 上提升了多少？", "ground_truth": "Llama 2 的上下文長度增加到了 4096 tokens (Llama 1 是 2048)。"},
    {"question": "QLoRA 技術引入了哪種數據類型來進一步壓縮模型？", "ground_truth": "4-bit NormalFloat (NF4)。"},
    {"question": "Llama 2-Chat 模型使用了哪種技術來提高人類偏好的一致性？", "ground_truth": "使用了 RLHF (Reinforcement Learning from Human Feedback) 和 Rejection Sampling。"},
    {"question": "Adapter Tuning 與 LoRA 的主要區別是什麼？", "ground_truth": "Adapter Tuning 通常在層之間插入新的模組，會增加推理延遲；而 LoRA 可以與原權重合併，不會增加推理延遲。"},
    {"question": "Soft Prompts (Prompt Tuning) 的做法是什麼？", "ground_truth": "不改變模型參數，而是在輸入序列前加上可學習的向量 (Learnable Vectors)。"},
    {"question": "Llama 模型家族通常使用什麼激活函數 (Activation Function)？", "ground_truth": "SwiGLU 激活函數。"},
    {"question": "在 PEFT 中，「Prefix Tuning」是針對模型的哪個部分進行操作？", "ground_truth": "針對每一層 Transformer 的 Key 和 Value 增加可訓練的 Prefix。"},

    # --- 4. ChatGPT Survey ---
    {"question": "AIGC 的全名是什麼？", "ground_truth": "AI-Generated Content (人工智慧生成內容)。"},
    {"question": "根據 Survey，ChatGPT 的核心架構是基於哪一個 GPT 版本？", "ground_truth": "GPT-3.5 (或 InstructGPT 的變體)。"},
    {"question": "ChatGPT 在訓練過程中引入了哪個關鍵步驟來減少有害輸出？", "ground_truth": "RLHF (Reinforcement Learning from Human Feedback)。"},
    {"question": "論文中提到 Generative AI 的三大主要模態通常是指哪三種？", "ground_truth": "Text (文本), Image (圖像), Audio (音頻)。"},
    {"question": "什麼是「Zero-shot Learning」在 GPT 模型中的定義？", "ground_truth": "模型在沒有看過任何特定任務範例的情況下，直接完成任務的能力。"},
    {"question": "Survey 中提到 ChatGPT 目前面臨的一個主要倫理問題是什麼？", "ground_truth": "偏見 (Bias)、版權問題 (Copyright) 或 生成虛假信息 (Misinformation)。"},
    {"question": "Chain-of-Thought (CoT) 提示工程的主要作用是什麼？", "ground_truth": "引導模型逐步推理，將複雜問題拆解成步驟，以提高推理準確率。"},
    {"question": "ChatGPT 的「知識截止日期 (Knowledge Cutoff)」意味著什麼？", "ground_truth": "模型無法獲知訓練數據截止日期之後發生的事件或資訊。"},
    {"question": "論文中提到 GPT-3 的參數量是多少？", "ground_truth": "175 Billion (1750億)。"},
    {"question": "InstructGPT 相比 GPT-3，主要的改進點是什麼？", "ground_truth": "更能遵循使用者的指令 (Better instruction following) 且更有幫助、更真實、更無害。"},

    # --- 5. Gemini 2.5 Technical Report (已補上數據) ---
    {"question": "Gemini 2.5 在「長上下文 (Long Context)」方面，支援的最大長度是多少？", "ground_truth": "支援高達 100 萬 (1M) 到 200 萬 (2M) tokens 的上下文窗口。"},
    {"question": "Gemini 2.5 提到的 'Agentic Capabilities' 指的是什麼能力？", "ground_truth": "模型能夠自主規劃、執行多步驟任務，並在環境中持續行動 (如玩遊戲或操作電腦)。"},
    {"question": "Gemini 2.5 在多模態 (Multimodality) 上有什麼顯著的影片處理能力？", "ground_truth": "它能夠處理長達 3 小時的影片內容，並理解其中的時間軸與細節。"},
    {"question": "相較於 Gemini 1.5，Gemini 2.5 在推理 (Reasoning) 基準測試上的表現如何？", "ground_truth": "它在 GPQA Diamond (科學問答) 和 MATH 基準測試上達到了最先進 (SoTA) 的水準。"},
    {"question": "Gemini 2.5 是否使用了 MoE (Mixture-of-Experts) 架構？", "ground_truth": "是的，它使用了稀疏混合專家 (Sparse MoE) 架構來提升訓練與推論效率。"},
    {"question": "文件中有提到 Gemini 2.5 針對哪種特定任務進行了優化？", "ground_truth": "針對程式碼生成 (Coding) 與複雜邏輯推理進行了深度優化。"},
    {"question": "Gemini 2.5 在處理「跨模態檢索」時，能做到什麼程度的精細度？", "ground_truth": "它能做到「像素級」或「幀級 (Frame-level)」的理解，精確定位影片或圖片中的特定物件。"},
    {"question": "關於「幻覺率 (Hallucination Rate)」，Gemini 2.5 有什麼改善？", "ground_truth": "透過強化學習 (RLHF) 和事實查核工具的整合，顯著降低了長文本生成中的幻覺。"},
    {"question": "Gemini 2.5 的訓練數據是否包含了影片內容？", "ground_truth": "是的，它是一個原生多模態 (Natively Multimodal) 模型，從預訓練階段就包含影片和音頻數據。"},
    {"question": "Gemini 2.5 在 MMLU (Massive Multitask Language Understanding) 測試集上的表現如何？", "ground_truth": "得分超過 90% (具體視版本而定，通常優於 GPT-4o)。"}
]

# 存檔
file_path = "golden_dataset.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ 已成功建立 {file_path}，包含 {len(data)} 題測試資料！")