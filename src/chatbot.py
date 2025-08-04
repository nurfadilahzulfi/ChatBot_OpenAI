from typing import List, Dict, Any, Optional
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from src.vector_store import VectorStoreManager
from src.retriever import DocumentRetriever
from config.settings import settings


class RAGChatbot:
    """Main RAG Chatbot class"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.CHAT_MODEL,
            temperature=0.7
        )
        
        self.vector_store_manager = VectorStoreManager()
        self.retriever = DocumentRetriever(self.vector_store_manager)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self.conversation_chain = None
        self._setup_conversation_chain()
    
    def _setup_conversation_chain(self):
        """Setup the conversational retrieval chain"""
        # Custom prompt template
        prompt_template = """
        Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan dokumen yang disediakan.
        Gunakan konteks berikut untuk menjawab pertanyaan pengguna dengan akurat dan informatif.
        
        Konteks dari dokumen:
        {context}
        
        Riwayat percakapan:
        {chat_history}
        
        Pertanyaan: {question}
        
        Instruksi:
        1. Jawab pertanyaan berdasarkan konteks yang diberikan
        2. Jika informasi tidak tersedia dalam konteks, katakan bahwa Anda tidak memiliki informasi tersebut
        3. Berikan jawaban yang jelas dan mudah dipahami
        4. Sertakan referensi ke sumber dokumen jika relevan
        
        Jawaban:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        try:
            base_retriever = self.vector_store_manager.get_retriever()
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=base_retriever,
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": prompt},
                return_source_documents=True,
                verbose=True
            )
        except ValueError as e:
            print(f"Warning: Could not setup conversation chain - {e}")
    
    def chat(self, message: str, retrieval_method: str = "similarity") -> Dict[str, Any]:
        """Process user message and return response"""
        if not self.conversation_chain:
            return {
                "answer": "Maaf, sistem belum siap. Pastikan dokumen telah dimuat ke dalam vector store.",
                "source_documents": [],
                "sources": []
            }
        
        try:
            # Get response from conversation chain
            response = self.conversation_chain({"question": message})
            
            # Extract sources
            sources = []
            if "source_documents" in response:
                sources = self.retriever.get_document_sources(response["source_documents"])
            
            return {
                "answer": response["answer"],
                "source_documents": response.get("source_documents", []),
                "sources": sources
            }
        
        except Exception as e:
            print(f"Error during chat: {e}")
            return {
                "answer": f"Maaf, terjadi kesalahan: {str(e)}",
                "source_documents": [],
                "sources": []
            }
    
    def get_relevant_context(self, query: str, method: str = "similarity") -> str:
        """Get relevant context for a query"""
        documents = self.retriever.retrieve_documents(query, method)
        return self.retriever.format_retrieved_context(documents)
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        print("Conversation memory cleared")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        messages = self.memory.chat_memory.messages
        history = []
        
        for message in messages:
            if isinstance(message, HumanMessage):
                history.append({"type": "human", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"type": "ai", "content": message.content})
        
        return history
    
    def add_documents_to_knowledge_base(self, documents):
        """Add new documents to the knowledge base"""
        self.vector_store_manager.add_documents(documents)
        # Reinitialize conversation chain with updated vector store
        self._setup_conversation_chain()
        print("Knowledge base updated successfully")
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search documents and return formatted results"""
        documents = self.vector_store_manager.similarity_search(query, k=k)
        
        results = []
        for doc in documents:
            results.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "metadata": doc.metadata
            })
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get chatbot statistics"""
        return {
            "vector_store_type": settings.VECTOR_STORE_TYPE,
            "embedding_model": settings.EMBEDDING_MODEL,
            "chat_model": settings.CHAT_MODEL,
            "chunk_size": settings.CHUNK_SIZE,
            "retrieval_k": settings.RETRIEVAL_K,
            "conversation_length": len(self.memory.chat_memory.messages)
        }