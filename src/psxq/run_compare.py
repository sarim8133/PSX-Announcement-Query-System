from __future__ import annotations
import argparse
import json
from pathlib import Path
from psxq.corpus import load_corpus
from psxq.vector.compare import compare_arms

def main() -> None:
    ap = argparse.ArgumentParser(description="Compare structured spine vs vector-RAG.")
    ap.add_argument("--records", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--today", required=True)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()
    records = load_corpus(args.records)
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    print(json.dumps(compare_arms(gold, records, today=args.today, k=args.k), indent=2))

if __name__ == "__main__":
    main()
