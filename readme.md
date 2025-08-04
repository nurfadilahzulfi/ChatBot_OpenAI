# RAG Chatbot dengan LangChain

Chatbot berbasis Retrieval-Augmented Generation (RAG) yang menggunakan LangChain dan OpenAI API untuk menjawab pertanyaan berdasarkan dokumen yang Anda upload.

## 🚀 Fitur

- **Multi-format Document Support**: PDF, TXT, JSON, DOCX, CSV
- **Advanced RAG Pipeline**: Document loading, chunking, embedding, dan retrieval
- **Multiple Vector Stores**: ChromaDB dan FAISS
- **Flexible Retrieval Methods**: Similarity search, compression, dan hybrid
- **Interactive Web Interface**: Streamlit-based UI
- **Conversation Memory**: Menyimpan konteks percakapan
- **Document Management**: Upload, scan, dan manage dokumen
- **Real-time Statistics**: Monitor performa sistem

## 📁 Struktur Project

```
rag_chatbot/ <br>
├── app.py                          # Main Streamlit application <br>
├── requirements.txt                # Python dependencies <br>
├── .env                           # Environment variables <br>
├── README.md                      # Documentation <br>
├── config/
│   └── settings.py                # Configuration settings <br>
├── data/ <br>
│   ├── documents/                 # Source documents <br>
│   │   ├── pdf/                  # PDF files <br>
│   │   ├── json/                 # JSON files <br>
│   │   ├── txt/                  # Text files<br>
│   │   ├── docx/                 # Word documents<br>
│   │   └── csv/                  # CSV files<br>
│   └── vectorstore/              # Vector database storage
├── src/
│   ├── document_loader.py        # Document loading utilities
│   ├── text_processor.py         # Text processing and chunking
│   ├── vector_store.py           # Vector store management
│   ├── retriever.py              # Document retrieval logic
│   └── chatbot.py                # Main chatbot logic
├── utils/
│   └── helpers.py                # Helper functions
└── tests/
    └── test_chatbot.py           # Unit tests
```

## 🛠️ Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd rag_chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables

Buat file `.env` di root directory:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Vector Store Configuration
VECTOR_STORE_TYPE=chroma  # options: chroma, faiss
PERSIST_DIRECTORY=./data/vectorstore

# Embedding Model
EMBEDDING_MODEL=text-embedding-ada-002

# Chat Model
CHAT_MODEL=gpt-3.5-turbo

# Chunk Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval Configuration
RETRIEVAL_K=4
```

### 4. Setup Directory Structure

```bash
python -c "from utils.helpers import create_directory_structure; create_directory_structure()"
```

## 🚀 Usage

### 1. Jalankan Aplikasi

```bash
streamlit run app.py
```

### 2. Upload Dokumen

- Letakkan dokumen Anda di folder yang sesuai:
  - PDF: `data/documents/pdf/`
  - TXT: `data/documents/txt/`
  - JSON: `data/documents/json/`
  - DOCX: `data/documents/docx/`
  - CSV: `data/documents/csv/`

### 3. Load Dokumen ke Vector Store

- Klik "🔍 Scan Dokumen" di sidebar
- Klik "📥 Load Dokumen ke Vector Store"
- Tunggu proses embedding selesai

### 4. Mulai Chat

- Ketik pertanyaan Anda di chat input
- Bot akan menjawab berdasarkan dokumen yang telah di-load
- Source dokumen akan ditampilkan untuk referensi

## 🔧 Konfigurasi

### Vector Store Options

**ChromaDB (Recommended)**
- Persistent storage
- Optimal untuk development
- Built-in metadata filtering

**FAISS**
- High performance
- Good untuk production
- Requires manual persistence

### Retrieval Methods

1. **Similarity**: Basic cosine similarity search
2. **Compression**: LLM-based context compression
3. **Hybrid**: Kombinasi multiple methods

### Model Configuration

- **Embedding Model**: `text-embedding-ada-002` (default)
- **Chat Model**: `gpt-3.5-turbo` atau `gpt-4`
- **Chunk Size**: 1000 characters (adjustable)
- **Chunk Overlap**: 200 characters (adjustable)

## 📚 API Usage

### Programmatic Usage

```python
from src.chatbot import RAGChatbot
from src.document_loader import DocumentLoader
from src.text_processor import TextProcessor

# Initialize components
chatbot = RAGChatbot()
loader = DocumentLoader()
processor = TextProcessor()

# Load and process documents
documents = loader.load_documents()
processed_