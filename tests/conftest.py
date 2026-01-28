"""
Pytest configuration and fixtures for the test suite.
"""

import sys
import pytest
from unittest.mock import patch, MagicMock
import os
from langchain_core.runnables import Runnable

# Set test environment variables
os.environ['OPENROUTER_API_KEY'] = 'test-key'
os.environ['LLM_MODEL_ID'] = 'test-model'


class MockRunnable(Runnable):
    """A proper Runnable mock for testing."""
    
    def __init__(self, name="mock"):
        super().__init__()
        self.name = name
    
    def invoke(self, input_data, config=None):
        return {"output": f"mock response from {self.name}", "input": input_data}
    
    def batch(self, inputs, config=None, **kwargs):
        return [self.invoke(inp, config) for inp in inputs]
    
    async def ainvoke(self, input_data, config=None):
        return self.invoke(input_data, config)
    
    async def abatch(self, inputs, config=None, **kwargs):
        return [self.invoke(inp, config) for inp in inputs]


def pytest_configure(config):
    """Mock langchain modules before any imports."""
    # Create a real Tool-like class that can be used with mocks
    class MockTool:
        def __init__(self, name=None, func=None, description=None):
            self.name = name
            self.func = func
            self.description = description
    
    # Create mock modules for all langchain submodules
    mock_modules = {
        'langchain': MagicMock(),
        'langchain.agents': MagicMock(),
        'langchain.chains': MagicMock(),
        'langchain.chat_models': MagicMock(),
        'langchain.memory': MagicMock(),
        'langchain.tools': MagicMock(),
        'langchain.embeddings': MagicMock(),
        'langchain.vectorstores': MagicMock(),
    }
    
    # Add the Tool class to the langchain.tools mock
    mock_modules['langchain.tools'].Tool = MockTool
    
    for module_name, mock_module in mock_modules.items():
        sys.modules[module_name] = mock_module


@pytest.fixture(autouse=True)
def mock_dotenv():
    """Auto-mock dotenv.load_dotenv to avoid loading .env file during tests."""
    with patch('dotenv.load_dotenv'):
        yield


@pytest.fixture
def mock_openai_embeddings():
    """Mock OpenAI embeddings to avoid API calls."""
    with patch('app.retriever.OpenAIEmbeddings') as mock:
        yield mock


@pytest.fixture
def mock_faiss():
    """Mock FAISS to avoid loading vector store."""
    with patch('app.retriever.FAISS') as mock:
        # Mock the load_local and as_retriever chain
        mock_faiss_instance = MagicMock()
        mock_retriever = MagicMock()
        mock_faiss_instance.as_retriever.return_value = mock_retriever
        mock.load_local.return_value = mock_faiss_instance
        yield mock


@pytest.fixture
def mock_chat_openai():
    """Mock ChatOpenAI to avoid API calls."""
    with patch('app.llm.ChatOpenAI') as mock:
        yield mock


@pytest.fixture
def mock_retrieval_qa():
    """Mock RetrievalQA to avoid complex initialization."""
    with patch('app.chains.RetrievalQA') as mock:
        yield mock
