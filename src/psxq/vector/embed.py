from __future__ import annotations
from psxq.llm import get_client

_EMBED_MODEL = "gemini-embedding-001"

def embed_text(text: str, client=None, model: str = _EMBED_MODEL) -> list[float]:
    client = client or get_client()
    resp = client.models.embed_content(model=model, contents=text)
    return list(resp.embeddings[0].values)
