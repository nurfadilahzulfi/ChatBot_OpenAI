import os
from typing import List, Optional
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from config.settings import settings


class VectorStoreManager:
    """Manages vector store operations"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.EMBEDDING_MODEL
        )
        self.vector_store = None
        self._load_or_create_vector_store()
    
    def _load_or_create_vector_store(self):
        """Load existing vector store or create new one"""
        if settings.VECTOR_STORE_TYPE.lower() == "chroma":
            self._setup_chroma()
        elif settings.VECTOR_STORE_TYPE.lower() == "faiss":
            self._setup_faiss()
        else:
            raise ValueError(f"Unsupported vector store type: {settings.VECTOR_STORE_TYPE}")
    
    def _setup_chroma(self):
        """Setup ChromaDB vector store"""
        if os.path.exists(settings.PERSIST_DIRECTORY) and os.listdir(settings.PERSIST_DIRECTORY):
            # Load existing vector store
            self.vector_store = Chroma(
                persist_directory=settings.PERSIST_DIRECTORY,
                embedding_function=self.embeddings
            )
            print("Loaded existing ChromaDB vector store")
        else:
            # Create new vector store (will be populated later)
            self.vector_store = None
            print("ChromaDB vector store will be created when documents are added")
    
    def _setup_faiss(self):
        """Setup FAISS vector store"""
        faiss_path = os.path.join(settings.PERSIST_DIRECTORY, "faiss_index")
        if os.path.exists(f"{faiss_path}.faiss"):
            # Load existing vector store
            self.vector_store = FAISS.load_local(faiss_path, self.embeddings)
            print("Loaded existing FAISS vector store")
        else:
            # Create new vector store (will be populated later)
            self.vector_store = None
            print("FAISS vector store will be created when documents are added")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to vector store"""
        if not documents:
            print("No documents to add")
            return
        
        if self.vector_store is None:
            # Create new vector store
            if settings.VECTOR_STORE_TYPE.lower() == "chroma":
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=settings.PERSIST_DIRECTORY
                )
                self.vector_store.persist()
            elif settings.VECTOR_STORE_TYPE.lower() == "faiss":
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
                self._save_faiss()
        else:
            # Add to existing vector store
            if settings.VECTOR_STORE_TYPE.lower() == "chroma":
                self.vector_store.add_documents(documents)
                self.vector_store.persist()
            elif settings.VECTOR_STORE_TYPE.lower() == "faiss":
                self.vector_store.add_documents(documents)
                self._save_faiss()
        
        print(f"Added {len(documents)} documents to vector store")
    
    def _save_faiss(self):
        """Save FAISS vector store"""
        faiss_path = os.path.join(settings.PERSIST_DIRECTORY, "faiss_index")
        self.vector_store.save_local(faiss_path)
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Search for similar documents"""
        if self.vector_store is None:
            print("Vector store is empty")
            return []
        
        k = k or settings.RETRIEVAL_K
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = None) -> List[tuple]:
        """Search for similar documents with similarity scores"""
        if self.vector_store is None:
            print("Vector store is empty")
            return []
        
        k = k or settings.RETRIEVAL_K
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def get_retriever(self, search_kwargs: dict = None):
        """Get retriever for the vector store"""
        if self.vector_store is None:
            raise ValueError("Vector store is empty")
        
        search_kwargs = search_kwargs or {"k": settings.RETRIEVAL_K}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def delete_documents(self, ids: List[str] = None):
        """Delete documents from vector store"""
        if self.vector_store is None:
            print("Vector store is empty")
            return
        
        if settings.VECTOR_STORE_TYPE.lower() == "chroma":
            if ids:
                self.vector_store.delete(ids=ids)
            else:
                # Delete all documents
                self.vector_store.delete_collection()
            self.vector_store.persist()
        else:
            print("Delete operation not implemented for FAISS")
    
    def reset_vector_store(self):
        """Reset the vector store"""
        if settings.VECTOR_STORE_TYPE.lower() == "chroma":
            if self.vector_store:
                self.vector_store.delete_collection()
        elif settings.VECTOR_STORE_TYPE.lower() == "faiss":
            faiss_path = os.path.join(settings.PERSIST_DIRECTORY, "faiss_index")
            if os.path.exists(f"{faiss_path}.faiss"):
                os.remove(f"{faiss_path}.faiss")
                os.remove(f"{faiss_path}.pkl")
        
        self.vector_store = None
        print("Vector store reset")