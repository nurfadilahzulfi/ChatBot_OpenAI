import streamlit as st
from src.chatbot import RAGChatbot
from src.document_loader import DocumentLoader
from src.text_processor import TextProcessor
from config.settings import settings

# Sisa logic streamlit seperti sebelumnya
import os
import time
from src.document_loader import DocumentLoader
from src.text_processor import TextProcessor
from src.chatbot import RAGChatbot
from utils.helpers import (
    create_directory_structure,
    scan_documents_directory,
    get_file_info,
    log_message
)
from config.settings import settings


def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'processing' not in st.session_state:
        st.session_state.processing = False


def setup_sidebar():
    """Setup sidebar with configuration and controls"""
    st.sidebar.title("⚙️ Konfigurasi RAG Chatbot")
    
    # API Key check
    api_key_status = "✅ Configured" if settings.OPENAI_API_KEY else "❌ Not Set"
    st.sidebar.write(f"**OpenAI API Key:** {api_key_status}")
    
    # Vector Store Info
    st.sidebar.write(f"**Vector Store:** {settings.VECTOR_STORE_TYPE}")
    st.sidebar.write(f"**Embedding Model:** {settings.EMBEDDING_MODEL}")
    st.sidebar.write(f"**Chat Model:** {settings.CHAT_MODEL}")
    
    st.sidebar.divider()
    
    # Document Management
    st.sidebar.subheader("📁 Manajemen Dokumen")
    
    # Scan documents
    if st.sidebar.button("🔍 Scan Dokumen"):
        with st.sidebar:
            with st.spinner("Scanning documents..."):
                files = scan_documents_directory(settings.DATA_DIR)
                st.session_state.document_files = files
                st.success(f"Found {len(files)} documents")
    
    # Load documents
    if st.sidebar.button("📥 Load Dokumen ke Vector Store"):
        if not settings.OPENAI_API_KEY:
            st.sidebar.error("Please set OpenAI API key first!")
            return
        
        load_documents_to_vector_store()
    
    # Clear vector store
    if st.sidebar.button("🗑️ Reset Vector Store"):
        if st.session_state.chatbot:
            st.session_state.chatbot.vector_store_manager.reset_vector_store()
            st.session_state.chatbot = None
            st.session_state.documents_loaded = False
            st.sidebar.success("Vector store reset!")
    
    st.sidebar.divider()
    
    # Chat Controls
    st.sidebar.subheader("💬 Kontrol Chat")
    
    if st.sidebar.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        if st.session_state.chatbot:
            st.session_state.chatbot.clear_memory()
        st.rerun()
    
    # Retrieval method selection
    retrieval_method = st.sidebar.selectbox(
        "Metode Retrieval:",
        ["similarity", "compression", "hybrid"],
        index=0
    )
    st.session_state.retrieval_method = retrieval_method


def load_documents_to_vector_store():
    """Load documents into vector store"""
    try:
        st.session_state.processing = True
        
        # Create progress bar
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        # Initialize components
        status_text.text("Initializing components...")
        progress_bar.progress(10)
        
        document_loader = DocumentLoader()
        text_processor = TextProcessor()
        
        # Load documents
        status_text.text("Loading documents...")
        progress_bar.progress(30)
        
        documents = document_loader.load_documents()
        
        if not documents:
            st.sidebar.warning("No documents found to load!")
            return
        
        # Process documents
        status_text.text("Processing and chunking documents...")
        progress_bar.progress(50)
        
        processed_docs = text_processor.process_documents(documents)
        
        # Initialize chatbot and add documents
        status_text.text("Creating vector embeddings...")
        progress_bar.progress(70)
        
        if not st.session_state.chatbot:
            st.session_state.chatbot = RAGChatbot()
        
        st.session_state.chatbot.add_documents_to_knowledge_base(processed_docs)
        
        # Complete
        progress_bar.progress(100)
        status_text.text("Completed!")
        
        st.session_state.documents_loaded = True
        st.sidebar.success(f"Loaded {len(processed_docs)} document chunks!")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.sidebar.error(f"Error loading documents: {str(e)}")
        log_message(f"Error loading documents: {str(e)}", "ERROR")
    
    finally:
        st.session_state.processing = False


def display_document_info():
    """Display information about loaded documents"""
    if hasattr(st.session_state, 'document_files'):
        st.subheader("📋 Dokumen yang Tersedia")
        
        files = st.session_state.document_files
        if files:
            # Create columns for file info
            cols = st.columns([3, 1, 1, 2])
            cols[0].write("**Filename**")
            cols[1].write("**Type**")
            cols[2].write("**Size**")
            cols[3].write("**Modified**")
            
            for file_info in files:
                cols = st.columns([3, 1, 1, 2])
                cols[0].write(file_info['name'])
                cols[1].write(file_info['extension'])
                cols[2].write(file_info['size_formatted'])
                cols[3].write(file_info['modified'].strftime("%Y-%m-%d"))
        else:
            st.info("No documents found. Please add documents to the data/documents folder.")


def display_chat_interface():
    """Display main chat interface"""
    st.subheader("💬 RAG Chatbot")
    
    if not settings.OPENAI_API_KEY:
        st.error("Please set your OpenAI API key in the .env file")
        return
    
    if not st.session_state.documents_loaded:
        st.info("Please load documents first using the sidebar controls.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.write(f"- {source}")
    
    # Chat input
    if prompt := st.chat_input("Tanyakan sesuatu tentang dokumen Anda..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if not st.session_state.chatbot:
                    st.session_state.chatbot = RAGChatbot()
                
                retrieval_method = getattr(st.session_state, 'retrieval_method', 'similarity')
                response = st.session_state.chatbot.chat(prompt, retrieval_method)
                
                st.markdown(response["answer"])
                
                # Display sources
                if response["sources"]:
                    with st.expander("📚 Sources"):
                        for source in response["sources"]:
                            st.write(f"- {source}")
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"]
        })


def display_statistics():
    """Display system statistics"""
    if st.session_state.chatbot:
        stats = st.session_state.chatbot.get_statistics()
        
        st.subheader("📊 Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Vector Store", stats["vector_store_type"])
            st.metric("Chat Model", stats["chat_model"])
        
        with col2:
            st.metric("Embedding Model", stats["embedding_model"])
            st.metric("Chunk Size", stats["chunk_size"])
        
        with col3:
            st.metric("Retrieval K", stats["retrieval_k"])
            st.metric("Conversation Length", stats["conversation_length"])


def main():
    """Main application function"""
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG Chatbot dengan LangChain")
    st.markdown("Chatbot yang menggunakan Retrieval-Augmented Generation untuk menjawab pertanyaan berdasarkan dokumen Anda.")
    
    # Create directory structure
    create_directory_structure()
    
    # Initialize session state
    initialize_session_state()
    
    # Setup sidebar
    setup_sidebar()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📁 Documents", "📊 Statistics"])
    
    with tab1:
        display_chat_interface()
    
    with tab2:
        display_document_info()
    
    with tab3:
        display_statistics()


if __name__ == "__main__":
    main()