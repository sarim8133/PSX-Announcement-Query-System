from psxq.llm import call_model

class _FakeResp:
    text = '{"ok": true}'

class _FakeModels:
    def __init__(self): self.last = None
    def generate_content(self, **kwargs):
        self.last = kwargs
        return _FakeResp()

class _FakeClient:
    def __init__(self): self.models = _FakeModels()

def test_call_model_passes_temperature_and_returns_text():
    client = _FakeClient()
    out = call_model("hello", client=client)
    assert out == '{"ok": true}'
    assert client.models.last["config"]["temperature"] == 1.0
