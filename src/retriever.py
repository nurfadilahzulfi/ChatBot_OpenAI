from typing import List, Dict, Any
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.llms import OpenAI
from src.vector_store import VectorStoreManager
from config.settings import settings


class DocumentRetriever:
    """Handles document retrieval with various strategies"""
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
        self.base_retriever = None
        self.compression_retriever = None
        self._setup_retrievers()
    
    def _setup_retrievers(self):
        """Setup different types of retrievers"""
        try:
            # Base retriever
            self.base_retriever = self.vector_store_manager.get_retriever()
            
            # Compression retriever for better context
            llm = OpenAI(
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=0
            )
            compressor = LLMChainExtractor.from_llm(llm)
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.base_retriever
            )
        except ValueError as e:
            print(f"Warning: Could not setup retrievers - {e}")
    
    def retrieve_documents(self, query: str, method: str = "similarity") -> List[Document]:
        """Retrieve relevant documents using specified method"""
        if not self.base_retriever:
            print("No retriever available")
            return []
        
        try:
            if method == "similarity":
                return self._similarity_retrieval(query)
            elif method == "compression":
                return self._compression_retrieval(query)
            elif method == "hybrid":
                return self._hybrid_retrieval(query)
            else:
                raise ValueError(f"Unknown retrieval method: {method}")
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
    
    def _similarity_retrieval(self, query: str) -> List[Document]:
        """Basic similarity search retrieval"""
        return self.base_retriever.get_relevant_documents(query)
    
    def _compression_retrieval(self, query: str) -> List[Document]:
        """Retrieval with contextual compression"""
        if not self.compression_retriever:
            return self._similarity_retrieval(query)
        
        return self.compression_retriever.get_relevant_documents(query)
    
    def _hybrid_retrieval(self, query: str) -> List[Document]:
        """Hybrid retrieval combining multiple methods"""
        # Get similarity results
        similarity_docs = self._similarity_retrieval(query)
        
        # Get documents with scores for filtering
        scored_docs = self.vector_store_manager.similarity_search_with_score(query)
        
        # Filter by score threshold (adjust as needed)
        score_threshold = 0.7
        filtered_docs = [doc for doc, score in scored_docs if score < score_threshold]
        
        # Combine and deduplicate
        all_docs = similarity_docs + filtered_docs
        unique_docs = self._deduplicate_documents(all_docs)
        
        return unique_docs[:settings.RETRIEVAL_K]
    
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents based on content"""
        seen_content = set()
        unique_docs = []
        
        for doc in documents:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs
    
    def retrieve_with_metadata_filter(self, query: str, metadata_filter: Dict[str, Any]) -> List[Document]:
        """Retrieve documents with metadata filtering"""
        # Get all relevant documents first
        all_docs = self.retrieve_documents(query)
        
        # Filter by metadata
        filtered_docs = []
        for doc in all_docs:
            match = True
            for key, value in metadata_filter.items():
                if key not in doc.metadata or doc.metadata[key] != value:
                    match = False
                    break
            if match:
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def get_document_sources(self, documents: List[Document]) -> List[str]:
        """Extract unique sources from documents"""
        sources = set()
        for doc in documents:
            if "source" in doc.metadata:
                sources.add(doc.metadata["source"])
        return list(sources)
    
    def format_retrieved_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string"""
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content.strip()
            context_parts.append(f"[Document {i} - {source}]\n{content}\n")
        
        return "\n".join(context_parts)