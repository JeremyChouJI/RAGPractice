# 🤖 Mini Gemini RAG Project

> 一個基於 Google Gemini 2.5 與 ChromaDB 的檢索增強生成 (RAG) 系統，具備 OCR 處理與完整的後端架構。

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)
![Gemini](https://img.shields.io/badge/Model-Gemini%202.5-purple)

## 📖 專案簡介 (Introduction)

本分支( `refactor/langchain-backend` )專注於使用 LangChain 框架建構 RAG (Retrieval-Augmented Generation) 的演示腳本 (Demo Scripts)。

設計核心在於展示 RAG 的高層次架構與運作原理。因此，實作上最大程度地採用了 LangChain 的原生模組，不另行撰寫底層的客製化邏輯，旨在提供一個標準、清晰且易於理解的 RAG 流程範例。

## 🛠️ 技術棧 (Tech Stack)

* **LLM Model**: Google Gemini-2.5-Flash / Pro
* **Embedding**: Google Generative AI Embeddings (`text-embedding-004`)
* **Vector DB**: ChromaDB (Local Persistence)
* **Backend Framework**: FastAPI
* **Orchestration**: LangChain
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
├── evaluation/         # 評估模型
├── requirements.txt    # 專案依賴
└── .env                # 環境變數設定
```
## 🧴 瓶頸 (Bottle Neck)

```text
- 