import csv
from types import SimpleNamespace
from app.sheet_tools.categorize import categorize_csv


def test_categorize_csv_basic(tmp_path, monkeypatch):
    # Prepare input CSV
    csv_path = tmp_path / "input.csv"
    content = "item,description\napple,red\nchair,wood\n"
    csv_path.write_text(content, encoding="utf-8")

    # Prepare mocked LLM response (as CSV) including the index column
    response_csv = "__row_index,item,description,category\n0,apple,red,Fruit\n1,chair,wood,Not Fruit\n"

    # Mock LLM client
    mock_llm = SimpleNamespace(invoke=lambda messages: SimpleNamespace(content=response_csv))
    monkeypatch.setattr('app.sheet_tools.categorize.get_llm', lambda llm_type='openrouter': mock_llm)

    # Run categorization
    result = categorize_csv(str(csv_path), categories=['Fruit', 'Not Fruit'], columns=['item', 'description'])

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]['category'] == 'Fruit'
    assert result[1]['category'] == 'Not Fruit'


def test_categorize_csv_saves_output(tmp_path, monkeypatch):
    csv_path = tmp_path / "input.csv"
    content = "item,description\napple,red\n"
    csv_path.write_text(content, encoding="utf-8")

    response_csv = "__row_index,item,description,category\n0,apple,red,Fruit\n"
    mock_llm = SimpleNamespace(invoke=lambda messages: SimpleNamespace(content=response_csv))
    monkeypatch.setattr('app.sheet_tools.categorize.get_llm', lambda llm_type='openrouter': mock_llm)

    out_file = tmp_path / "out.csv"
    result = categorize_csv(str(csv_path), categories=['Fruit'], columns=['item', 'description'], output_path=str(out_file))

    assert out_file.exists()
    saved = out_file.read_text(encoding='utf-8')
    assert 'category' in saved
    assert result[0]['category'] == 'Fruit'


def test_categorize_csv_chunking(tmp_path, monkeypatch):
    # 3 rows, chunk size 2 -> expect two LLM invocations
    csv_path = tmp_path / "input.csv"
    content = "item,description\napple,red\nbanana,yellow\nchair,wood\n"
    csv_path.write_text(content, encoding="utf-8")

    responses = [
        "__row_index,item,description,category\n0,apple,red,Fruit\n1,banana,yellow,Fruit\n",
        "__row_index,item,description,category\n2,chair,wood,Not Fruit\n",
    ]

    call_state = {"n": 0}

    def invoke(messages):
        idx = call_state["n"]
        call_state["n"] += 1
        return SimpleNamespace(content=responses[idx])

    mock_llm = SimpleNamespace(invoke=invoke)
    monkeypatch.setattr('app.sheet_tools.categorize.get_llm', lambda llm_type='openrouter': mock_llm)

    result = categorize_csv(str(csv_path), categories=['Fruit', 'Not Fruit'], columns=['item', 'description'], max_rows=2)

    assert len(result) == 3
    assert result[0]['category'] == 'Fruit'
    assert result[1]['category'] == 'Fruit'
    assert result[2]['category'] == 'Not Fruit'
    assert call_state['n'] == 2
