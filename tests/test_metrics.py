from psxq.eval.metrics import (doc_set_metrics, recall_at_k, hit_at_k, mrr,
                               schema_coverage, aggregate)

def test_doc_set_metrics_exact_and_prf():
    m = doc_set_metrics(["A", "B"], ["A", "B"])
    assert m["exact"] is True and m["precision"] == 1.0 and m["recall"] == 1.0
    m2 = doc_set_metrics(["A", "B"], ["A", "C"])
    assert m2["exact"] is False and m2["precision"] == 0.5 and m2["recall"] == 0.5

def test_doc_set_metrics_empty_gold_and_pred_is_exact():
    m = doc_set_metrics([], [])
    assert m["exact"] is True

def test_recall_and_hit_at_k():
    assert recall_at_k(["A", "B", "C"], ["B"], k=2) == 1.0
    assert hit_at_k(["A", "B", "C"], ["Z"], k=2) == 0.0

def test_mrr_rank_position():
    assert mrr(["A", "B", "C"], ["B"]) == 0.5
    assert mrr(["A", "B"], ["Z"]) == 0.0

def test_schema_coverage_counts_fractions():
    cov = schema_coverage(["structured", "structured", "schema_blocked", "semantic"])
    assert cov["structured"]["count"] == 2
    assert cov["schema_blocked"]["frac"] == 0.25

def test_aggregate_mean_min_max():
    a = aggregate([2.0, 4.0, 3.0])
    assert a["mean"] == 3.0 and a["min"] == 2.0 and a["max"] == 4.0
