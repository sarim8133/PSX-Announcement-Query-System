from psxq.corpus import load_corpus
from psxq.vector.rag import build_index, rag_answer

RECS = load_corpus("data/synthetic/records.json")

def _fake_embed(text, client=None):
    # toy embedding: presence of keywords -> 3-dim vector
    t = text.lower()
    return [float("dividend" in t), float("closed period" in t or "closure" in t),
            float("right issue" in t)]

def test_build_index_embeds_each_record(monkeypatch):
    monkeypatch.setattr("psxq.vector.rag.embed_text", _fake_embed)
    idx = build_index(RECS)
    assert set(idx._ids) == {"S001", "S002", "S003", "S004", "S005"}

def test_rag_answer_retrieves_then_generates(monkeypatch):
    monkeypatch.setattr("psxq.vector.rag.embed_text", _fake_embed)
    monkeypatch.setattr("psxq.vector.rag.generate_answer",
                        lambda q, records, client=None: "GEN:" + ",".join(r.doc_id for r in records))
    idx = build_index(RECS)
    res = rag_answer("any dividend news?", RECS, idx, k=2)
    assert "S002" in res["doc_ids"]      # dividend record retrieved
    assert res["answer"].startswith("GEN:")
