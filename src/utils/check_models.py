import os
import google.generativeai as genai

# 記得設環境變數
# os.environ["GOOGLE_API_KEY"] = "..."

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("📋 Your API key can be used with the following list of models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")