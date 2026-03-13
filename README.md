# 👾 全端 RAG 檢索增強生成系統 (Full-stack RAG System)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)
![Gemini](https://img.shields.io/badge/Model-Gemini%202.5-purple)
![REPL](https://img.shields.io/badge/Interface-REPL-4EAA25?logo=gnu-bash&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-cc5500)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white)

## 📖 專案簡介 (Overview)
本專案開發了一套具備意圖路由 (Intent Routing) 功能的 AI Agent 系統，旨在透過 LLM 的推理能力解決複雜的使用者需求。系統採用模組化設計，將 RAG 封裝為核心工具之一，專門處理領域知識檢索。

## 🛠️ 核心功能與技術

### 進階檢索 (Advanced RAG Implementation)
- Hybrid Search:
    - 實作結合「關鍵字檢索 (Keyword Search)」與「語意向量檢索 (Semantic Vector Search)」的混合搜尋演算法。
    - 有效解決單一向量檢索在面對專有名詞或精確匹配時的準確度不足問題。
- Re-ranking:
    - 整合 Cross-Encoder 進行二次精確排序，優化檢索結果的排列順序。
    - 提升檢索系統在複雜問答場景下的準確性，確保輸入模型的 Context 具備最高關聯度。

### 系統工程與部署 (System Engineering & Deployment)
- Docker 部署：
    - 編寫 Dockerfile 建立標準化執行環境，解決 "It works on my machine" 的問題。
    - 使用 docker-compose 進行服務編排，實現一鍵啟動 (One-click deployment) 與服務管理（如 docker compose run --rm 進行測試與除錯）。

- 開發維運 (DevOps)：
    - 實作環境變數管理 (.env)，確保 API Key 等敏感資訊與程式碼分離，考慮資安問題。
    - 設定 .gitignore 進行版控過濾，確保推送到 GitHub 的程式碼庫乾淨且安全。

### 程式開發與架構 (Development & Architecture)
- 工具: 
    - **LLM Framework**: LangChain
    - **LLM Model**: Google Gemini-2.5-Flash / Gemini-2.5-Pro
    - **Embedding**: Google Generative AI Embeddings (`text-embedding-004`)
    - **Document Loader**: LangChain Community Loaders
    - **Vector DB**: ChromaDB (Local Persistence)
    - **Backend Framework**: FastAPI
    - **Frontend**: Vanilla JS + HTML/CSS

- 模組化設計： 將資料處理、檢索邏輯與生成模組解耦，便於未來整合不同的 LLM 模型或 Vector Database。

### 📂 專案架構 (Project Structure)

```text
rag-project/
├── src/                    # 核心邏輯 (Backend & AI Agent)
│   └── tool/               # AI Agent 模型封裝
├── static/               # 簡易使用者介面層 (User Interface)
├── evaluation/             # RAG 效果評估模組 (用於測試檢索準確率與回答品質)
├── .env.example            # 環境變數範本 (隱藏敏感資訊，資安考量)
├── docker-compose.yaml     # 服務編排設定 (定義 Agent 與其他服務的連動)
├── Dockerfile              # Image建置
├── entrypoint.sh           # Container 啟動腳本
└── requirements.txt        # Python 相依套件清單
```