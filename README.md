# 🎯RAGPractice — Retrieval-Augmented Generation Playground

> 使用 Python、LangChain、Google Gemini 打造的 RAG 練習專案。  
> 支援 PDF、CSV、OCR、Chroma DB、Metadata Filter、Score Threshold 等進階功能。  
> 本專案想從 0 到進階、循序學習 RAG 的開發者。

---

## 📌 Features

- **🔍 PDF / CSV 解析**
  - 支援一般 PDF 與掃描 PDF（內建 OCR via Tesseract）
  - CSV 可選擇語意搜尋或工具模式（Tool Calling）

- **🧩 Chunking & Embedding**
  - 使用 `RecursiveCharacterTextSplitter`
  - 自訂 chunk size / overlap
  - 使用 `GoogleGenerativeAIEmbeddings`（可替換）

- **📚 Chroma Vector Store**
  - 本地向量資料庫
  - 自動 metadata：檔名、頁碼、來源類型等

- **🎯 Retrieval Quality 提升**
  - Score threshold（濾掉不相關片段）
  - Metadata filter（只查特定檔案或類型）
  - 動態 top-k 策略

- **💬 RAG 問答引擎**
  - 基於 Gemini 2.0 / 2.5 Pro
  - 自動組 Prompt + context
  - 嚴格遵守「資料沒有就說不知道」

- **📦 Modules 可擴充**
  - 加入 FastAPI、Docker、RAG 評估、Snippet 高亮等功能

---
## 🧠RAG Workflow

```css
[Load Documents] → [Chunk] → [Embed] → [Vector Store]
        ↑                                      ↓
        └────────── [Retriever] ← Question ← [LLM]
```
        
RAG（Retrieval-Augmented Generation）借助外部知識庫來降低 LLM 的幻覺並提升回答正確性。

---

## 🛠️ 主要技術棧 (Tech Stack)
| 類別            | 技術                    |
| ------------- | --------------------- |
| **LLM**       | Google Gemini         |
| **Embedding** | text-embedding-004    |
| **Vector DB** | ChromaDB              |
| **OCR**       | Tesseract + pdf2image |
| **Parsing**   | PyPDF                 |
| **Framework** | LangChain             |
| **Language**  | Python 3.11 / 3.12    |

---

## 📝 Roadmap

- ⬜ Score threshold 自動調整

- ⬜ CSV Tool Mode（結構化查詢 Router）

- ✅ 多輪對話支援 + Snippet 高亮

- ⬜ RAG 評估工具（不同 chunk size / k 表現）

- ✅ FastAPI inference API

- ⬜ Docker 化

- ⬜ 雲端部署（GCP / AWS）

## 📝 Note
- 如果要使用 OCR 記得需要安裝 POPPLER + Tesseract, 並設定環境變數
  - https://github.com/UB-Mannheim/tesseract/wiki
  - https://github.com/oschwartz10612/poppler-windows/releases/
