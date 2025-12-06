import os
import pytesseract
from typing import List, Dict, Any
from pypdf import PdfReader
from pdf2image import convert_from_path
from langchain_core.documents import Document

#Have to set enviroment variables first
POPPLER_PATH = os.environ.get("POPPLER_PATH")
TESSERACT_CMD = os.environ.get("TESSERACT_CMD")

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

class pdf_loader:
    def _load_pdf(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    def load_pdfs_from_folder(self, folder_path: str) -> str:
        all_text = ""
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(folder_path, filename)
                print(f"📄 Loading PDF：{filename}")
                all_text += self._load_pdf(pdf_path) + "\n"
        return all_text
    
    def load_pdfs_from_folder(self, folder_path: str) -> List[Document]:
        all_docs: List[Document] = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if not os.path.isfile(file_path):
                continue

            lower = filename.lower()
            if lower.endswith(".pdf"):
                all_docs.extend(self.load_pdf_as_documents(file_path))
            else:
                continue

        return all_docs
    
    def _ocr_pdf_page(self, file_path: str, page_number_1based: int, lang: str = "chi_tra+eng") -> str:
        images = convert_from_path(
            file_path,
            first_page=page_number_1based,
            last_page=page_number_1based,
            poppler_path=POPPLER_PATH,
        )
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img, lang=lang)
        return text

    def load_pdf_as_documents(self, file_path: str) -> List[Document]:
        print(f"📄Loading PDF：{os.path.basename(file_path)}")

        try:
            reader = PdfReader(file_path)
        except Exception as e:
            #print(f"⚠️  Couldn't load PDF：{file_path}，error：{e}")
            return []

        docs: List[Document] = []
        filename = os.path.basename(file_path)

        for i, page in enumerate(reader.pages):
            page_number = i + 1  # 人類看的頁碼從 1 開始
            text = ""

            try:
                text = page.extract_text() or ""
            except Exception as e:
                #print(f"⚠️ Failed to extract text on page {page_number} , attempting OCR instead. error：{e}")
                pass

            if not text.strip():
                try:
                    #print(f"Using OCR...")
                    text = self._ocr_pdf_page(file_path, page_number)
                except Exception as e:
                    #print(f"⚠️  OCR also failed on page {page_number}: {e}")
                    text = ""

            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": page_number,
                            "file_path": file_path,
                            "type": "pdf",
                        },
                    )
                )
            else:
                #print(f"⚠️  Page {page_number} contains no extractable text at all, skipping.")
                pass

        return docs