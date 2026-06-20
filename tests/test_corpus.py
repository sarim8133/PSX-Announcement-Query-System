from psxq.corpus import load_corpus
from psxq.models import Record

def test_load_corpus_returns_records():
    recs = load_corpus("data/synthetic/records.json")
    assert len(recs) == 5
    assert all(isinstance(r, Record) for r in recs)
    assert recs[0].doc_id == "S001"
    assert recs[1].financials.payout_amount == 7.5
