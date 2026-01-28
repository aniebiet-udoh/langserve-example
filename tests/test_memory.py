"""
Unit tests for the memory module.
"""

import pytest
from unittest.mock import patch, MagicMock
from app.memory import get_memory


class TestMemory:
    """Tests for the memory.py module."""
    
    def test_get_memory_returns_conversation_buffer_memory(self):
        """Test that get_memory returns a ConversationBufferMemory instance."""
        with patch('app.memory.ConversationBufferMemory') as mock_memory:
            mock_instance = MagicMock()
            mock_memory.return_value = mock_instance
            
            result = get_memory()
            
            assert result == mock_instance
            mock_memory.assert_called_once()
    
    def test_get_memory_configures_memory_key(self):
        """Test that get_memory configures memory with correct key."""
        with patch('app.memory.ConversationBufferMemory') as mock_memory:
            mock_instance = MagicMock()
            mock_memory.return_value = mock_instance
            
            get_memory()
            
            # Verify it was called with memory_key="history"
            call_kwargs = mock_memory.call_args[1]
            assert call_kwargs.get('memory_key') == 'history'
    
    def test_get_memory_returns_messages_enabled(self):
        """Test that get_memory enables return_messages."""
        with patch('app.memory.ConversationBufferMemory') as mock_memory:
            mock_instance = MagicMock()
            mock_memory.return_value = mock_instance
            
            get_memory()
            
            # Verify it was called with return_messages=True
            call_kwargs = mock_memory.call_args[1]
            assert call_kwargs.get('return_messages') is True
