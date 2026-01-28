"""
Unit tests for the agent module.
"""

import pytest
from unittest.mock import patch, MagicMock
from app.agent import get_agent


class TestGetAgent:
    """Tests for the get_agent factory function."""
    
    def test_get_agent(self):
        agent = get_agent()
        assert agent is not None
        print("Agent created successfully:", agent)
        assert isinstance(agent, str)
