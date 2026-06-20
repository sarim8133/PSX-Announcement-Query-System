from psxq.vector.index import VectorIndex

def test_search_ranks_by_cosine():
    idx = VectorIndex()
    idx.add("A", [1.0, 0.0])
    idx.add("B", [0.0, 1.0])
    idx.add("C", [0.9, 0.1])
    ranked = idx.search([1.0, 0.0], k=2)
    assert ranked == ["A", "C"]

def test_search_k_limits_results():
    idx = VectorIndex()
    for i, v in enumerate([[1, 0], [0.8, 0.2], [0, 1]]):
        idx.add(f"D{i}", v)
    assert len(idx.search([1, 0], k=1)) == 1
