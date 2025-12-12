# 🤖 Mini Gemini RAG Project

> 一個基於 Google Gemini 2.5 與 ChromaDB 的檢索增強生成 (RAG) 系統，具備 OCR 處理與完整的後端架構。

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)
![Gemini](https://img.shields.io/badge/Model-Gemini%202.5-purple)

## 📖 專案簡介 (Introduction)

這個專案是一個輕量級但架構完整的 RAG (Retrieval-Augmented Generation) 實作。旨在解決 LLM 無法存取私有數據及幻覺 (Hallucination) 問題。

不同於常見的 Demo 腳本，本專案採用**分層架構 (Layered Architecture)** 設計，將資料處理 (Ingestion)、模型邏輯 (Model)、與 API 服務分離，並針對實際應用中常見的 **PDF 解析困難 (Dirty Data)** 問題實作了 OCR 容錯機制。

### ✨ 核心功能 (Key Features)

* **⚡ 高效能 LLM 整合**：串接 **Google Gemini-2.5-Flash**，利用其長文本優勢處理複雜 Context。
* **👁️ 強健的 PDF 解析 (Robust Parsing)**：
    * 使用 `pypdf` 進行初步提取。
    * **OCR Fallback 機制**：當偵測到掃描檔或無法提取文字的頁面時，自動切換至 `Tesseract OCR` 進行光學辨識，確保資料召回率 (Recall)。
* **🗄️ 持久化向量資料庫**：使用 **ChromaDB** 儲存 Embeddings，實現資料持久化，無需重複計算向量。
* **🏗️ 模組化架構**：清晰分離前端、API 層與 RAG 核心邏輯，易於維護與擴充。
* **🔍 精確檢索**：實作 Metadata Filtering (依檔名/類型過濾) 與信心分數門檻 (Score Threshold) 過濾。

## 🛠️ 技術棧 (Tech Stack)

* **LLM Model**: Google Gemini-2.5-Flash / Pro
* **Embedding**: Google Generative AI Embeddings (`text-embedding-004`)
* **Vector DB**: ChromaDB (Local Persistence)
* **Backend Framework**: FastAPI
* **Orchestration**: LangChain
* **PDF/OCR**: `pypdf`, `pdf2image`, `pytesseract` (Tesseract-OCR)
* **Frontend**: Vanilla JS + HTML/CSS

## 📂 專案架構 (Project Structure)

```text
ragTutorial/
├── src/
│   ├── api/            # FastAPI 路由與進入點
│   ├── models/         # RAG 核心邏輯 (Retriever, ChatSession)
│   └── utils/          # 工具函式 (PDF Loader, OCR 處理)
├── data_source/        # 放置 PDF 文件的目錄
├── frontend/           # 簡易 Web 介面
├── requirements.txt    # 專案依賴
└── .env                # 環境變數設定
