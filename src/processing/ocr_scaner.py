import os
import glob
from tqdm import tqdm
from langchain_community.document_loaders import UnstructuredPDFLoader

def batch_convert_ocr(source_folder, output_folder):
    """
    Docstring for batch_convert_ocr
    以OCR掃描.pdf輸出成.txt，用以後續資料預處理
    
    :param source_folder: 輸入PDF位址
    :param output_folder: 輸出.txt位址
    """
    if not os.path.exists(source_folder):
        print(f"❌ Source directory not found: {source_folder}")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 Created output directory: {output_folder}")

    pdf_files = glob.glob(os.path.join(source_folder, "*.pdf"))
    total_files = len(pdf_files)
    
    if total_files == 0:
        print("⚠️ No PDF files found in the source directory.")
        return

    print(f"🚀 Found {total_files} PDFs. Starting batch OCR conversion...")
    print("-" * 50)

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        filename = os.path.basename(pdf_path)

        try:
            loader = UnstructuredPDFLoader(
                file_path=pdf_path,
                mode="elements",
                strategy="fast", #純文字用fast就好了
                languages=["eng"]
            )
            
            raw_docs = loader.load()
            if not raw_docs:
                print(f"   ⚠️ Warning: No text extracted from {filename}")
                continue
            full_text = "\n\n".join([doc.page_content for doc in raw_docs])

            txt_filename = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join(output_folder, txt_filename)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            
            print(f"   ✅ Saved to: {txt_filename} (Length: {len(full_text)})")

        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")

    print("-" * 50)
    print("🎉 All done!")

if __name__ == "__main__":
    input_dir = "./data_source" 
    output_dir = "./txt_output"
    
    batch_convert_ocr(input_dir, output_dir)