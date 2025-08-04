from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from config.settings import settings


class TextProcessor:
    """Handles text processing and chunking"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process and chunk documents"""
        print(f"Processing {len(documents)} documents...")
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk.page_content)
            })
        
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove special characters if needed
        # text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
        
        return text.strip()
    
    def preprocess_document(self, document: Document) -> Document:
        """Preprocess a single document"""
        # Clean the content
        cleaned_content = self.clean_text(document.page_content)
        
        # Update the document
        document.page_content = cleaned_content
        
        return document