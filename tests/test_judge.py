from psxq.eval.judge import judge_answer

def test_judge_parses_correct_and_grounded(monkeypatch):
    monkeypatch.setattr("psxq.eval.judge.call_model",
                        lambda prompt, client=None: '{"correct": true, "grounded": true}')
    out = judge_answer("q", ["Beta Cement Limited", "7.5"], "Beta Cement pays 7.5", "context")
    assert out == {"correct": True, "grounded": True}

def test_judge_handles_fenced_json(monkeypatch):
    monkeypatch.setattr("psxq.eval.judge.call_model",
                        lambda prompt, client=None: '```json\n{"correct": false, "grounded": true}\n```')
    out = judge_answer("q", ["x"], "wrong", "ctx")
    assert out["correct"] is False
