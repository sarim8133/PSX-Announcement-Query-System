from __future__ import annotations
import json
import time
from psxq.llm import call_model
from psxq.planner import strip_fences

JUDGE_PROMPT = """You are scoring an answer about PSX announcements.
Question: {question}
Required facts (gold): {facts}
Retrieved context the answer was allowed to use:
{context}
Answer to score:
{answer}

Return JSON only:
{{"correct": <true if the answer states all required gold facts and contradicts none>,
  "grounded": <true if every claim in the answer is supported by the retrieved context>}}
[nonce:{nonce}]"""

def judge_answer(question: str, gold_facts, answer: str, context: str, client=None) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, facts=gold_facts, context=context,
                                 answer=answer, nonce=time.time_ns())
    raw = call_model(prompt, client=client)
    cleaned = strip_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Extract true/false values directly when JSON is malformed
        import re
        correct = bool(re.search(r'"correct"\s*:\s*true', cleaned, re.IGNORECASE))
        grounded = bool(re.search(r'"grounded"\s*:\s*true', cleaned, re.IGNORECASE))
        return {"correct": correct, "grounded": grounded}
    return {"correct": bool(data["correct"]), "grounded": bool(data["grounded"])}
