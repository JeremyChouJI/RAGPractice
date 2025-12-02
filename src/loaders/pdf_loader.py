import os
from pypdf import PdfReader
class pdf_loader:
    def load_pdf(self, file_path: str) -> str:
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
                all_text += self.load_pdf(pdf_path) + "\n"
        return all_text