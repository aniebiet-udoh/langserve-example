"""
Unit tests for the LLM module.
"""

import pytest
import os
from unittest.mock import patch, MagicMock, call
from app.llm import get_llm


class TestGetLLM:
    """Tests for the get_llm factory function."""

    def test_get_llm(self):
        result = get_llm()
        assert result is not None