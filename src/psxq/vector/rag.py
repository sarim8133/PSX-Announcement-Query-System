from __future__ import annotations
from psxq.models import Record
from psxq.generator import generate_answer
from psxq.vector.embed import embed_text
from psxq.vector.index import VectorIndex

def _doc_text(r: Record) -> str:
    return f"{r.company_name}. {r.subject}. signals: {', '.join(r.announcement_signals)}. {r.summary}"

def build_index(records: list[Record], client=None) -> VectorIndex:
    idx = VectorIndex()
    for r in records:
        idx.add(r.doc_id, embed_text(_doc_text(r), client=client))
    return idx

def rag_answer(question: str, records: list[Record], index: VectorIndex,
               k: int = 4, client=None) -> dict:
    by_id = {r.doc_id: r for r in records}
    qvec = embed_text(question, client=client)
    ranked = index.search(qvec, k=k)
    selected = [by_id[d] for d in ranked if d in by_id]
    answer = generate_answer(question, selected, client=client)
    return {"doc_ids": ranked, "answer": answer}
