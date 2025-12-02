🎯 RAGPractice — Retrieval-Augmented Generation Playground

使用 Python + LangChain + Google Gemini 所打造的 RAG 練習專案
支援 PDF、CSV、多模態 OCR、Chroma DB、Metadata Filter、Score Threshold 等功能。
這是一個從 0 到進階、可逐步擴充的 RAG 學習場域。

📌 專案特色 (Features)

🔍 PDF / CSV 解析

支援文字型 PDF

支援掃描 PDF（內建 OCR via Tesseract）

CSV 可用語意搜尋，或使用「工具模式 (Tool Calling)」解析結構化資料

🧩 Chunking & Embedding

使用 RecursiveCharacterTextSplitter

支援調整 chunk size / overlap

可替換 embedding（目前使用 GoogleGenerativeAIEmbeddings）

📚 Chroma Vector Store

本地向量資料庫

自動 metadata 紀錄（檔案名稱、頁碼、資料來源）

🎯 改善檢索品質

Score Threshold（過濾低相關 chunk）

Metadata Filter（只查特定類型資料：PDF / CSV / 指定檔名）

Top-k 動態調整

💬 RAG 問答引擎

用 Gemini 2.0 / 2.5 Pro 回答

遵守「不可亂編」規則

自動組合 context + 提示詞

📦 可擴充架構

之後可新增：FastAPI API、Docker、評估工具、snippet 高亮

🏗️ 專案結構 (Project Structure)
ragTutorial/
│
├── src/
│   ├── mini_gemini_rag.py           # 最小可用 RAG
│   ├── adv_mini_pdf_rag.py          # 進階版：OCR + metadata + threshold
│   ├── utils/                       # 工具模組 (optional)
│   └── ...
│
├── data_source/
│   ├── *.pdf                        # PDF 原始資料
│   ├── *.csv                        # CSV 原始資料
│   └── ...
│
├── requirements.txt
├── .gitignore
└── README.md

🚀 如何開始使用 (Getting Started)
1️⃣ 建立虛擬環境（建議）
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell

2️⃣ 安裝套件
pip install -r requirements.txt

3️⃣ 設定 API Key

在系統環境變數加入：

GOOGLE_API_KEY=你的API金鑰


或在 .env（已 gitignore）加入：

GOOGLE_API_KEY=xxxx

4️⃣ 執行 RAG 互動程式
python src/mini_gemini_rag.py


或進階版：

python src/adv_mini_pdf_rag.py

🧠 RAG 流程簡介
[Load Documents] → [Chunk] → [Embed] → [Vector Store]
        ↑                                      ↓
        └────────── [Retriever] ← Question ← [LLM]


本專案採用 Retrieval-Augmented Generation，避免 LLM 幻覺、提升回答品質。

🛠️ 主要技術棧 (Tech Stack)
類別	技術
LLM	Google Gemini
Embedding	text-embedding-004
Vector DB	ChromaDB
OCR	Tesseract + pdf2image
Parsing	PyPDF
Framework	LangChain
Language	Python 3.11 / 3.12
📝 未來 Roadmap

 Score threshold 自動化調整

 CSV Tool Mode（數據查詢路由器）

 多輪對話 + 引用 snippet 高亮

 RAG 評估工具（不同 chunk size / k）

 FastAPI 推論 API

 Docker 化

 上傳雲端 (GCP / AWS)

🤝 貢獻 (Contributing)

歡迎提出 Issue 或 PR，一起打造更完整的 RAG 學習專案！

📜 License

MIT License
