import re

def clean_hp_text(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    patterns = [
        r'^\s*\d+\s*$',                  # 單獨一行的頁碼
        r'^\s*[A-Z’\s]+(?:\s+\d+)?\s*$', # 全大寫標題及可能的後續數字
    ]
    
    # 合併&執行所有正則表達式
    combined_pattern = re.compile('|'.join(patterns), re.MULTILINE)
    cleaned_content = combined_pattern.sub('', content)
    # 將所有換行符號替換為一個空格
    cleaned_content = re.sub(r'\n+', ' ', cleaned_content)
    # 刪除重複空格
    cleaned_content = re.sub(r' {2,}', ' ', cleaned_content)
    # 清理多餘的空行
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
    # 修正 dan- gling 這種錯誤。
    cleaned_content = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', cleaned_content)
    # 移除章節標題
    cleaned_content = re.sub(r'— CHAPTER [A-Z ]+ —', '', cleaned_content)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_content.strip())

    print(f"清理完成！已儲存至 {output_file}")

# 使用範例
OUTPUT_FILE = './cleaned_data/Harry Potter and the Chamber of Secrets.txt'
INPUT_FILE = './txt_output/Harry Potter and the Chamber of Secrets.txt'
clean_hp_text(INPUT_FILE, OUTPUT_FILE)