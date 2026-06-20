from __future__ import annotations
import os

_DEFAULT_MODEL = "gemini-2.5-flash"

def get_client():
    from google import genai
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def call_model(prompt: str, client=None, model: str = _DEFAULT_MODEL,
               temperature: float = 1.0) -> str:
    client = client or get_client()
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"temperature": temperature},
    )
    return resp.text
