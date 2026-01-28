"""
Unit tests for the chains module.
"""

import pytest
from unittest.mock import patch, MagicMock
from app.chains import get_rag_chain


class TestGetRAGChain:
    """Tests for the get_rag_chain factory function."""
    
    def test_get_rag_chain_creates_retrieval_qa(self, mock_retrieval_qa):
        """Test that get_rag_chain creates a RetrievalQA instance."""
        mock_instance = MagicMock()
        mock_retrieval_qa.from_chain_type.return_value = mock_instance
        
        result = get_rag_chain()
        
        assert result == mock_instance
    
    def test_get_rag_chain_uses_from_chain_type(self, mock_retrieval_qa):
        """Test that get_rag_chain uses from_chain_type factory method."""
        mock_instance = MagicMock()
        mock_retrieval_qa.from_chain_type.return_value = mock_instance
        
        get_rag_chain()
        
        # Verify from_chain_type was called
        mock_retrieval_qa.from_chain_type.assert_called_once()
    
    def test_get_rag_chain_passes_llm(self, mock_retrieval_qa):
        """Test that get_rag_chain passes LLM to RetrievalQA."""
        mock_instance = MagicMock()
        mock_retrieval_qa.from_chain_type.return_value = mock_instance
        
        with patch('app.chains.get_llm') as mock_get_llm:
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm
            
            get_rag_chain()
            
            # Verify llm was passed in call
            call_kwargs = mock_retrieval_qa.from_chain_type.call_args[1]
            assert 'llm' in call_kwargs
    
    def test_get_rag_chain_passes_retriever(self, mock_retrieval_qa):
        """Test that get_rag_chain passes retriever to RetrievalQA."""
        mock_instance = MagicMock()
        mock_retrieval_qa.from_chain_type.return_value = mock_instance
        
        with patch('app.chains.get_retriever') as mock_get_retriever:
            mock_retriever = MagicMock()
            mock_get_retriever.return_value = mock_retriever
            
            get_rag_chain()
            
            # Verify retriever was passed in call
            call_kwargs = mock_retrieval_qa.from_chain_type.call_args[1]
            assert 'retriever' in call_kwargs
    
    def test_get_rag_chain_uses_stuff_chain_type(self, mock_retrieval_qa):
        """Test that get_rag_chain uses 'stuff' chain type."""
        mock_instance = MagicMock()
        mock_retrieval_qa.from_chain_type.return_value = mock_instance
        
        get_rag_chain()
        
        # Verify chain_type='stuff' was passed
        call_kwargs = mock_retrieval_qa.from_chain_type.call_args[1]
        assert call_kwargs.get('chain_type') == 'stuff'
    
    def test_get_rag_chain_returns_chain(self, mock_retrieval_qa):
        """Test that get_rag_chain returns a chain object."""
        mock_instance = MagicMock()
        mock_retrieval_qa.from_chain_type.return_value = mock_instance
        
        result = get_rag_chain()
        
        assert result is not None
        assert result == mock_instance
