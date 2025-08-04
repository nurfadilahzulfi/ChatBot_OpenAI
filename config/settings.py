import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CHAT_MODEL = "gpt-3.5-turbo"
    EMBEDDING_MODEL = "text-embedding-ada-002"  # ✅ Model embedding yang benar
    VECTOR_STORE_TYPE = "chroma"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    RETRIEVAL_K = 4
    DATA_DIR = "data/documents"
    PDF_DIR = "data/documents"
    PERSIST_DIRECTORY = "data/vectorstore"

settings = Settings()