import sys
import types

from app.pdfconv.message_builder import MessageBuilder
from app.pdfconv.ai import PdfConverter


class DummyMsg:
    def __init__(self, content):
        self.content = content


class DummyLLM:
    def __init__(self):
        self.last = None

    def invoke(self, messages):
        # store the messages so tests can inspect them
        self.last = messages

        class R:
            pass

        r = R()
        r.content = "```csv\ncol1,col2\n1,2\n```"
        return r


def _ensure_humanmessage(monkeypatch):
    # Create fake langchain_core.messages.HumanMessage
    pkg = types.ModuleType("langchain_core")
    messages_mod = types.ModuleType("langchain_core.messages")
    messages_mod.HumanMessage = DummyMsg
    pkg.messages = messages_mod
    monkeypatch.setitem(sys.modules, "langchain_core", pkg)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", messages_mod)


def test_messagebuilder_structured_supported(monkeypatch):
    _ensure_humanmessage(monkeypatch)

    msg = MessageBuilder.build_message(
        prompt="hello",
        chunk_data=b"%PDF-",
        llm_type="groq",
        use_structured_messages=True,
        extract_text=False,
    )

    assert isinstance(msg, DummyMsg)
    assert isinstance(msg.content, list)


def test_messagebuilder_fallback_to_string(monkeypatch):
    _ensure_humanmessage(monkeypatch)

    msg = MessageBuilder.build_message(
        prompt="hello",
        chunk_data=b"%PDF-",
        llm_type="openai",
        use_structured_messages=True,
        extract_text=False,
    )

    assert isinstance(msg, DummyMsg)
    assert isinstance(msg.content, str)
    assert "data:application/pdf;base64" in msg.content


def test_messagebuilder_extract_text(monkeypatch):
    _ensure_humanmessage(monkeypatch)

    class FakePage:
        def extract_text(self):
            return "col1,col2\n1,2"

    class FakeReader:
        def __init__(self, stream):
            self.pages = [FakePage()]

    # Patch PdfUtils.extract_text to use our fake reader
    import app.pdfconv.utils as utils
    monkeypatch.setattr(utils.PdfUtils, "extract_text", lambda data: "col1,col2\n1,2")

    msg = MessageBuilder.build_message(
        prompt="hello",
        chunk_data=b"%PDF-",
        llm_type="openai",
        use_structured_messages=False,
        extract_text=True,
    )

    assert isinstance(msg, DummyMsg)
    assert "col1,col2" in (msg.content if isinstance(msg.content, str) else str(msg.content))


def test_pdfconverter_invoke(monkeypatch):
    # Ensure HumanMessage exists in langchain_core
    _ensure_humanmessage(monkeypatch)

    converter = PdfConverter()
    llm = DummyLLM()

    # Call internal conversion method (mimics a chunk)
    result = converter._convert_chunk(
        chunk_data=b"%PDF-",
        llm=llm,
        llm_type="openai",
        chunk_info="",
        is_first_chunk=True,
        remove_header_if_not_first=False,
        use_structured_messages=False,
        extract_text=False,
    )

    assert "col1,col2" in result
