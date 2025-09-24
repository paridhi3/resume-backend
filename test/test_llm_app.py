import pytest
import json
import types
from unittest.mock import MagicMock, patch
import llm_app

def test_extract_text_from_pdf_reads_text(monkeypatch):
    # Mock fitz.open and page.get_text
    class DummyPage:
        def get_text(self, mode):
            return "Page text"
    class DummyDoc:
        def __enter__(self): return [DummyPage(), DummyPage()]
        def __exit__(self, exc_type, exc_val, exc_tb):
            # This method is intentionally left empty because DummyDoc does not need to release any resources.
            pass
    dummy_file = MagicMock()
    dummy_file.read.return_value = b"pdfbytes"
    monkeypatch.setattr(llm_app.fitz, "open", lambda stream, filetype: DummyDoc())
    text = llm_app.extract_text_from_pdf(dummy_file)
    assert text == "Page text\nPage text"

def test_process_and_save_resume_parses_email_and_llm(monkeypatch, tmp_path):
    # Setup
    dummy_file = MagicMock()
    dummy_file.name = "resume1.pdf"
    resume_text = "John Doe\njohn@example.com\nPython, SQL"
    # Patch LLM response
    class DummyResponse:
        class Choices:
            class Message:
                content = '{"name": "John Doe", "skills": ["Python", "SQL"], "education": ["BSc"], "experience": ["Dev"], "summary": "A summary"}'
            message = Message()
        choices = [Choices()]
    monkeypatch.setattr(llm_app.client.chat.completions, "create", lambda **kwargs: DummyResponse())
    # Patch METADATA_FILE to tmp_path
    monkeypatch.setattr(llm_app, "METADATA_FILE", str(tmp_path / "metadata.json"))
    result = llm_app.process_and_save_resume(dummy_file, resume_text)
    assert result["email"] == "john@example.com"
    assert result["name"] == "John Doe"
    assert "Python" in result["skills"]
    # Check file written
    with open(llm_app.METADATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert dummy_file.name in data

def test_compute_similarity_returns_float(monkeypatch):
    # Patch embedding_model.encode and cosine_similarity
    monkeypatch.setattr(llm_app.embedding_model, "encode", lambda x: [[1, 2, 3]])
    monkeypatch.setattr(llm_app, "cosine_similarity", lambda a, b: [[0.75]])
    score = llm_app.compute_similarity("jd", "resume")
    assert isinstance(score, float)
    assert score == pytest.approx(75.0)

def test_parse_llm_json_valid_and_invalid():
    # Valid JSON
    text = '{"category": "Suitable", "final_score": 90, "reasons": ["Good skills"]}'
    result = llm_app.parse_llm_json(text)
    assert result["category"] == "Suitable"
    # JSON with code block
    text2 = "```json\n{\"category\": \"Maybe\", \"final_score\": 50, \"reasons\": [\"Missing skills\"]}\n```"
    result2 = llm_app.parse_llm_json(text2)
    assert result2["category"] == "Maybe"
    # Invalid JSON
    text3 = "Not a JSON"
    result3 = llm_app.parse_llm_json(text3)
    assert result3["category"] == "Error"

def test_analyze_with_llm_calls_openai(monkeypatch):
    # Patch client.chat.completions.create
    called = {}
    def dummy_create(**kwargs):
        called["prompt"] = kwargs["messages"][0]["content"]
        class DummyResp:
            class Choices:
                class Message:
                    content = '{"category": "Suitable", "final_score": 95, "reasons": ["Great match"]}'
                message = Message()
            choices = [Choices()]
        return DummyResp()
    monkeypatch.setattr(llm_app.client.chat.completions, "create", dummy_create)
    metadata = {
        "name": "Jane",
        "email": "jane@x.com",
        "skills": ["Python"],
        "experience": ["Dev"],
        "education": ["BSc"],
        "summary": "Summary"
    }
    result = llm_app.analyze_with_llm("JD text", metadata, 80)
    assert "Suitable" in result
    assert "95" in result
    assert "Great match" in result
    assert "JD text" in called["prompt"]