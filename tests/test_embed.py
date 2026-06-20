from psxq.vector.embed import embed_text

def test_embed_text_returns_vector():
    class _R:  # shape mirrors google-genai embed response
        embeddings = [type("E", (), {"values": [0.1, 0.2, 0.3]})()]
    class _M:
        def embed_content(self, **kwargs): return _R()
    class _C:
        models = _M()
    vec = embed_text("hello", client=_C())
    assert vec == [0.1, 0.2, 0.3]
