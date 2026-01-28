from langchain.tools import tool

@tool
def calculator(expression: str):
    """Evaluate mathematical expressions"""
    # One of those cases of a tool that does everything!
    try:
        return str(eval(expression))  # Very bad!!! No guardrails.
    except Exception as e:
        return f"Error: {e}"

@tool
def search_notes(query: str):
    """Search internal ML notes for relevant information"""
    return "Use retrieval chain for this query"


def get_tools():
    return [calculator, search_notes]
