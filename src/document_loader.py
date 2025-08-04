import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import json
from config.settings import settings

class DocumentLoader:
    """Handles loading documents from configured folder and subfolders"""

    def __init__(self):
        self.supported_extensions = [".pdf", ".txt", ".json"]
        self.folder_path = settings.PDF_DIR if hasattr(settings, 'PDF_DIR') else settings.DATA_DIR
        print(f"📁 Folder path yang digunakan: {self.folder_path}")

    def load_documents(self) -> List[Document]:
        documents = []

        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"📁 Folder {self.folder_path} tidak ditemukan.")

        # Scan semua file di folder dan subfolder
        for root, dirs, files in os.walk(self.folder_path):
            print(f"📂 Scanning folder: {root}")
            print(f"📄 Files found: {files}")
            
            for filename in files:
                file_path = os.path.join(root, filename)
                print(f"🔍 Memproses: {file_path}")

                ext = os.path.splitext(filename)[1].lower()
                print(f"📄 Ekstensi file: {ext}")

                if ext not in self.supported_extensions:
                    print(f"⏭️ Format tidak didukung: {filename} (ekstensi: {ext})")
                    continue

                try:
                    if ext == ".pdf":
                        print(f"📖 Loading PDF: {filename}")
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                    elif ext == ".txt":
                        print(f"📝 Loading TXT: {filename}")
                        loader = TextLoader(file_path, encoding="utf-8")
                        docs = loader.load()
                    elif ext == ".json":
                        print(f"📊 Loading JSON: {filename}")
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            text = self._flatten_json(data)
                            docs = [Document(page_content=text, metadata={"source": filename})]

                    print(f"✅ Berhasil load {len(docs)} dokumen dari {filename}")
                    documents.extend(docs)

                except Exception as e:
                    print(f"❌ Gagal load {filename}: {e}")

        print(f"📚 Total dokumen yang berhasil dimuat: {len(documents)}")
        return documents

    def _flatten_json(self, data: dict) -> str:
        """Ubah struktur JSON CPO ke bentuk teks yang bisa dipahami chatbot"""
        text = ""
        for tahun, bulan_data in data.items():
            for bulan, hari_data in bulan_data.items():
                for hari, detail in hari_data.items():
                    harga = detail.get("harga", "tidak diketahui")
                    text += f"Harga CPO pada {hari}-{bulan}-{tahun} adalah {harga}\n"
        return text