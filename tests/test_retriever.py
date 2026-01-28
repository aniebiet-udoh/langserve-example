"""
Unit tests for the retriever module.
"""

import pytest
from unittest.mock import patch, MagicMock
from app.retriever import get_retriever


class TestGetRetriever:
    """Tests for the get_retriever factory function."""
    
    def test_get_retriever_creates_embeddings(self, mock_openai_embeddings):
        """Test that get_retriever creates OpenAIEmbeddings."""
        with patch('app.retriever.FAISS') as mock_faiss:
            mock_faiss_instance = MagicMock()
            mock_faiss_instance.as_retriever.return_value = MagicMock()
            mock_faiss.load_local.return_value = mock_faiss_instance
            
            get_retriever()
            
            # Verify OpenAIEmbeddings was created
            mock_openai_embeddings.assert_called_once()
    
    def test_get_retriever_loads_faiss_from_vectorstore(self, mock_openai_embeddings):
        """Test that get_retriever loads FAISS from vectorstore directory."""
        with patch('app.retriever.FAISS') as mock_faiss:
            mock_faiss_instance = MagicMock()
            mock_faiss_instance.as_retriever.return_value = MagicMock()
            mock_faiss.load_local.return_value = mock_faiss_instance
            
            get_retriever()
            
            # Verify FAISS.load_local was called with correct path
            mock_faiss.load_local.assert_called_once()
            call_args = mock_faiss.load_local.call_args[0]
            assert 'vectorstore' in call_args
    
    def test_get_retriever_converts_to_retriever(self, mock_openai_embeddings):
        """Test that get_retriever converts FAISS DB to retriever."""
        with patch('app.retriever.FAISS') as mock_faiss:
            mock_faiss_instance = MagicMock()
            mock_retriever = MagicMock()
            mock_faiss_instance.as_retriever.return_value = mock_retriever
            mock_faiss.load_local.return_value = mock_faiss_instance
            
            result = get_retriever()
            
            # Verify as_retriever was called
            mock_faiss_instance.as_retriever.assert_called_once()
            assert result == mock_retriever
    
    def test_get_retriever_returns_retriever(self, mock_openai_embeddings):
        """Test that get_retriever returns a retriever object."""
        with patch('app.retriever.FAISS') as mock_faiss:
            mock_faiss_instance = MagicMock()
            mock_retriever = MagicMock()
            mock_faiss_instance.as_retriever.return_value = mock_retriever
            mock_faiss.load_local.return_value = mock_faiss_instance
            
            result = get_retriever()
            
            assert result is not None
            assert result == mock_retriever
