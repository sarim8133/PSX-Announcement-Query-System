from psxq.corpus import load_corpus
from psxq.generator import render_records, generate_answer

RECS = load_corpus("data/synthetic/records.json")

def test_render_records_includes_company_and_facts():
    text = render_records([RECS[1]])
    assert "Beta Cement Limited" in text
    assert "7.5" in text

def test_generate_answer_passes_records_into_prompt(monkeypatch):
    captured = {}
    def fake(prompt, client=None):
        captured["prompt"] = prompt
        return "Beta Cement Limited pays PKR 7.5/share."
    monkeypatch.setattr("psxq.generator.call_model", fake)
    out = generate_answer("dividends above 5?", [RECS[1]])
    assert "Beta Cement" in out
    assert "Beta Cement Limited" in captured["prompt"]

def test_generate_answer_empty_records_still_calls_model(monkeypatch):
    monkeypatch.setattr("psxq.generator.call_model",
                        lambda prompt, client=None: "No matching announcements found.")
    out = generate_answer("anything?", [])
    assert "No matching" in out
