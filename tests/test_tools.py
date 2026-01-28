"""
Unit tests for the tools module.
"""

import pytest
from app.tools import calculator, get_tools, calculator_tool, search_tool, search_notes


class TestCalculator:
    """Tests for the calculator tool function."""
    
    def test_calculator_addition(self):
        """Test calculator performs addition correctly."""
        result = calculator("2 + 2")
        assert result == "4"
    
    def test_calculator_subtraction(self):
        """Test calculator performs subtraction correctly."""
        result = calculator("10 - 3")
        assert result == "7"
    
    def test_calculator_multiplication(self):
        """Test calculator performs multiplication correctly."""
        result = calculator("5 * 6")
        assert result == "30"
    
    def test_calculator_division(self):
        """Test calculator performs division correctly."""
        result = calculator("20 / 4")
        assert result == "5.0"
    
    def test_calculator_complex_expression(self):
        """Test calculator handles complex expressions."""
        result = calculator("(10 + 5) * 2")
        assert result == "30"
    
    def test_calculator_returns_string(self):
        """Test that calculator returns result as string."""
        result = calculator("1 + 1")
        assert isinstance(result, str)


class TestCalculatorTool:
    """Tests for the calculator Tool object."""
    
    def test_calculator_tool_has_correct_name(self):
        """Test that calculator tool has correct name."""
        assert calculator_tool.name == "calculator"
    
    def test_calculator_tool_has_description(self):
        """Test that calculator tool has description."""
        assert calculator_tool.description == "Evaluate mathematical expressions"
    
    def test_calculator_tool_is_callable(self):
        """Test that calculator tool has func attribute."""
        assert callable(calculator_tool.func)
    
    def test_calculator_tool_func_works(self):
        """Test that calculator tool's func works correctly."""
        result = calculator_tool.func("3 + 3")
        assert result == "6"


class TestSearchNotes:
    """Tests for the search_notes function."""
    
    def test_search_notes_returns_string(self):
        """Test that search_notes returns a string."""
        result = search_notes("test query")
        assert isinstance(result, str)
    
    def test_search_notes_returns_expected_message(self):
        """Test that search_notes returns the expected message."""
        result = search_notes("anything")
        assert "retrieval chain" in result.lower()


class TestSearchTool:
    """Tests for the search_tool Tool object."""
    
    def test_search_tool_has_correct_name(self):
        """Test that search tool has correct name."""
        assert search_tool.name == "search_notes"
    
    def test_search_tool_has_description(self):
        """Test that search tool has description."""
        assert search_tool.description == "Search internal ML notes"
    
    def test_search_tool_is_callable(self):
        """Test that search tool has func attribute."""
        assert callable(search_tool.func)


class TestGetTools:
    """Tests for the get_tools factory function."""
    
    def test_get_tools_returns_list(self):
        """Test that get_tools returns a list."""
        result = get_tools()
        assert isinstance(result, list)
    
    def test_get_tools_returns_two_tools(self):
        """Test that get_tools returns exactly 2 tools."""
        result = get_tools()
        assert len(result) == 2
    
    def test_get_tools_includes_calculator(self):
        """Test that get_tools includes calculator tool."""
        result = get_tools()
        tool_names = [t.name for t in result]
        assert "calculator" in tool_names
    
    def test_get_tools_includes_search(self):
        """Test that get_tools includes search_notes tool."""
        result = get_tools()
        tool_names = [t.name for t in result]
        assert "search_notes" in tool_names
    
    def test_get_tools_returns_tool_objects(self):
        """Test that get_tools returns Tool objects."""
        result = get_tools()
        for tool in result:
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'func')
            assert hasattr(tool, 'description')
