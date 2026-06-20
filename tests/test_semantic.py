from psxq.corpus import load_corpus
from psxq.semantic import semantic_filter

RECS = load_corpus("data/synthetic/records.json")

def test_semantic_filter_keeps_ids_the_model_returns(monkeypatch):
    monkeypatch.setattr("psxq.semantic.call_model", lambda prompt, client=None: '["S005"]')
    out = semantic_filter("meetings where something was rejected", RECS)
    assert [r.doc_id for r in out] == ["S005"]

def test_semantic_filter_ignores_unknown_ids(monkeypatch):
    monkeypatch.setattr("psxq.semantic.call_model", lambda prompt, client=None: '["NOPE","S002"]')
    out = semantic_filter("x", RECS)
    assert [r.doc_id for r in out] == ["S002"]
