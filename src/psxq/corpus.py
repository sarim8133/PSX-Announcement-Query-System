from __future__ import annotations
import json
from pathlib import Path
from psxq.models import Record

def load_corpus(path: str | Path) -> list[Record]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Record(**r) for r in raw]
