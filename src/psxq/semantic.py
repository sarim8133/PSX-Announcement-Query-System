from __future__ import annotations
import json
import time
from psxq.llm import call_model
from psxq.models import Record
from psxq.planner import strip_fences

SEMANTIC_PROMPT = """Given a fuzzy information need and candidate announcements, return a JSON
array of the doc_ids that genuinely satisfy the need. Return only doc_ids present below.

Need: {intent}

Candidates:
{candidates}

Output ONLY a JSON array of doc_id strings, no markdown.
[nonce:{nonce}]"""

def _render(records: list[Record]) -> str:
    return "\n".join(f'{r.doc_id}: {r.subject} | {r.summary}' for r in records)

def semantic_filter(intent: str, candidates: list[Record], client=None) -> list[Record]:
    prompt = SEMANTIC_PROMPT.format(intent=intent, candidates=_render(candidates),
                                    nonce=time.time_ns())
    raw = call_model(prompt, client=client)
    keep = set(json.loads(strip_fences(raw)))
    return [r for r in candidates if r.doc_id in keep]
