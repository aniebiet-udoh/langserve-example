import csv
import types
import pytest

import app.pdfconvbasic as pdfconvbasic


class FakePage:
    def __init__(self, txt):
        self._txt = txt

    def extract_text(self):
        return self._txt


def test_text_output_prints_extracted_text(tmp_path, monkeypatch, capsys):
    pdf = tmp_path / "in.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    def Reader(data):
        return types.SimpleNamespace(pages=[FakePage("one"), FakePage("two")])

    monkeypatch.setattr(pdfconvbasic, "PyPDF2", types.SimpleNamespace(PdfReader=Reader))

    rc = pdfconvbasic.main([str(pdf), "--format", "text"])
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.out == "one\ntwo\n"


def test_csv_writes_file(tmp_path, monkeypatch):
    pdf = tmp_path / "in.pdf"
    out = tmp_path / "out.csv"
    pdf.write_bytes(b"%PDF-1.4")

    def Reader(data):
        return types.SimpleNamespace(pages=[FakePage("a"), FakePage("b\nc")])

    monkeypatch.setattr(pdfconvbasic, "PyPDF2", types.SimpleNamespace(PdfReader=Reader))

    rc = pdfconvbasic.main([str(pdf), "--format", "csv", "--output", str(out)])
    assert rc == 0

    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["page", "text"]
    assert rows[1] == ["1", "a"]
    assert rows[2] == ["2", "b c"]


def test_csv_dedupes_common_header(tmp_path, monkeypatch):
    pdf = tmp_path / "in.pdf"
    out = tmp_path / "out.csv"
    pdf.write_bytes(b"%PDF-1.4")

    hdr = "11 GRACE BILL ROAD, EKET\n"
    def Reader(data):
        return types.SimpleNamespace(pages=[FakePage(hdr + "page1 content"), FakePage(hdr + "page2 content")])

    monkeypatch.setattr(pdfconvbasic, "PyPDF2", types.SimpleNamespace(PdfReader=Reader))

    rc = pdfconvbasic.main([str(pdf), "--format", "csv", "--output", str(out)])
    assert rc == 0

    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[1] == ["1", "page1 content"]
    assert rows[2] == ["2", "page2 content"]


def test_csv_preserve_newlines_flag(tmp_path, monkeypatch):
    pdf = tmp_path / "in.pdf"
    out = tmp_path / "out.csv"
    pdf.write_bytes(b"%PDF-1.4")

    def Reader(data):
        return types.SimpleNamespace(pages=[FakePage("a\nb\nc")])

    monkeypatch.setattr(pdfconvbasic, "PyPDF2", types.SimpleNamespace(PdfReader=Reader))

    rc = pdfconvbasic.main([str(pdf), "--format", "csv", "--output", str(out), "--preserve-newlines"]) 
    assert rc == 0

    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    # newline preserved in the field
    assert rows[1][1] == "a\nb\nc"


def test_csv_requires_output(tmp_path):
    pdf = tmp_path / "in.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    with pytest.raises(SystemExit) as exc:
        pdfconvbasic.main([str(pdf), "--format", "csv"])  # missing --output

    assert exc.value.code != 0
