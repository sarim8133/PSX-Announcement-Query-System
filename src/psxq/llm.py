from __future__ import annotations
import os
import time

_DEFAULT_MODEL = "gemini-2.5-flash"

def get_client():
    from google import genai
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def call_model(prompt: str, client=None, model: str = _DEFAULT_MODEL,
               temperature: float = 1.0) -> str:
    client = client or get_client()
    last_err = None
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "thinking_config": {"thinking_budget": 0},
                },
            )
            return resp.text
        except Exception as e:
            msg = str(e).lower()
            if any(x in msg for x in ("429", "quota", "rate", "resource_exhausted", "503", "unavailable")):
                wait = (attempt + 1) * 15
                print(f"[llm] throttled (attempt {attempt+1}), retrying in {wait}s...")
                time.sleep(wait)
                last_err = e
            else:
                raise
    raise last_err
