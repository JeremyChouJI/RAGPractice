import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("正在查詢您的 API Key 可用的模型清單...")
try:
    for m in genai.list_models():
        if 'embedContent' in m.supported_generation_methods:
            print(f"找到支援 Embedding 的模型: {m.name}")
except Exception as e:
    print(f"無法獲取清單，錯誤訊息: {e}")