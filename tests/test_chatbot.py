import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from src.document_loader import DocumentLoader
from src.text_processor import TextProcessor
from src.vector_store import VectorStoreManager
from src.chatbot import RAGChatbot
from langchain.docstore.document import Document


class TestDocumentLoader(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.loader = DocumentLoader()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_load_txt_document(self):
        # Create a test text file
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("This is a test document.")
        
        documents = self.loader._load_txt(test_file)
        
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "This is a test document.")
        self.assertEqual(documents[0].metadata["source"], test_file)
    
    def test_supported_extensions(self):
        expected_extensions = {'.pdf', '.txt', '.json', '.docx', '.csv'}
        actual_extensions = set(self.loader.supported_extensions.keys())
        
        self.assertEqual(expected_extensions, actual_extensions)


class TestTextProcessor(unittest.TestCase):
    
    def setUp(self):
        self.processor = TextProcessor()
    
    def test_clean_text(self):
        dirty_text = "  This   is    a    test   text.  "
        clean_text = self.processor.clean_text(dirty_text)
        
        self.assertEqual(clean_text, "This is a test text.")
    
    def test_process_documents(self):
        # Create test documents
        documents = [
            Document(page_content="This is a test document. " * 100, metadata={"source": "test1.txt"}),
            Document(page_content="Another test document. " * 100, metadata={"source": "test2.txt"})
        ]
        
        processed_docs = self.processor.process_documents(documents)
        
        # Should create multiple chunks due to length
        self.assertGreater(len(processed_docs), len(documents))
        
        # Check metadata
        for doc in processed_docs:
            self.assertIn("chunk_id", doc.metadata)
            self.assertIn("chunk_size", doc.metadata)


class TestVectorStoreManager(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @patch('src.vector_store.OpenAIEmbeddings')
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-key',
        'VECTOR_STORE_TYPE': 'chroma',
        'PERSIST_DIRECTORY': None
    })
    def test_vector_store_initialization(self, mock_embeddings):
        # Mock embeddings
        mock_embeddings.return_value = MagicMock()
        
        with patch('config.settings.settings.PERSIST_DIRECTORY', self.temp_dir):
            manager = VectorStoreManager()
            
            self.assertIsNotNone(manager.embeddings)
    
    def test_similarity_search_empty_store(self):
        with patch('config.settings.settings.PERSIST_DIRECTORY', self.temp_dir):
            with patch('src.vector_store.OpenAIEmbeddings') as mock_embeddings:
                mock_embeddings.return_value = MagicMock()
                
                manager = VectorStoreManager()
                manager.vector_store = None
                
                results = manager.similarity_search("test query")
                self.assertEqual(len(results), 0)


class TestRAGChatbot(unittest.TestCase):
    
    @patch('src.chatbot.ChatOpenAI')
    @patch('src.chatbot.VectorStoreManager')
    @patch('src.chatbot.DocumentRetriever')
    def test_chatbot_initialization(self, mock_retriever, mock_vector_store, mock_llm):
        # Mock dependencies
        mock_llm.return_value = MagicMock()
        mock_vector_store.return_value = MagicMock()
        mock_retriever.return_value = MagicMock()
        
        chatbot = RAGChatbot()
        
        self.assertIsNotNone(chatbot.llm)
        self.assertIsNotNone(chatbot.vector_store_manager)
        self.assertIsNotNone(chatbot.retriever)
        self.assertIsNotNone(chatbot.memory)
    
    @patch('src.chatbot.ChatOpenAI')
    @patch('src.chatbot.VectorStoreManager')
    @patch('src.chatbot.DocumentRetriever')
    def test_chat_without_setup(self, mock_retriever, mock_vector_store, mock_llm):
        # Mock dependencies
        mock_llm.return_value = MagicMock()
        mock_vector_store.return_value = MagicMock()
        mock_retriever.return_value = MagicMock()
        
        chatbot = RAGChatbot()
        chatbot.conversation_chain = None
        
        response = chatbot.chat("Test message")
        
        self.assertIn("answer", response)
        self.assertIn("source_documents", response)
        self.assertIn("sources", response)
        self.assertIn("sistem belum siap", response["answer"].lower())


class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_document_processing_pipeline(self):
        # Create test document
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("This is a test document for integration testing. " * 50)
        
        # Test document loading
        loader = DocumentLoader()
        documents = loader._load_txt(test_file)
        
        self.assertEqual(len(documents), 1)
        
        # Test text processing
        processor = TextProcessor()
        processed_docs = processor.process_documents(documents)
        
        self.assertGreaterEqual(len(processed_docs), 1)
        
        # Check that chunks have proper metadata
        for doc in processed_docs:
            self.assertIn("chunk_id", doc.metadata)
            self.assertIn("source", doc.metadata)


if __name__ == '__main__':
    # Set test environment variables
    os.environ['OPENAI_API_KEY'] = 'test-key'
    os.environ['VECTOR_STORE_TYPE'] = 'chroma'
    os.environ['EMBEDDING_MODEL'] = 'text-embedding-ada-002'
    os.environ['CHAT_MODEL'] = 'gpt-3.5-turbo'
    
    unittest.main()