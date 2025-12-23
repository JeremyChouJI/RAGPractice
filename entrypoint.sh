#!/bin/bash

if [ ! -d "/app/chroma_db_eng" ] || [ -z "$(ls -A /app/chroma_db_eng)" ]; then
    echo "⚡ Database not found or empty detected; starting embedding process. "
    
    python -m src.utils.ingest_eng
    
    echo "✅ Embedding completed！"
else
    echo "👌 Existing database detected; skipping the embedding step."
fi

echo "🚀 Starting AI Agent..."
python -m src.AI_Agent