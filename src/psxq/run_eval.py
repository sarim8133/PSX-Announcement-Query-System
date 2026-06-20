from __future__ import annotations
import argparse
import json
from pathlib import Path
from psxq.corpus import load_corpus
from psxq.eval.harness import run_eval

def main() -> None:
    ap = argparse.ArgumentParser(description="Run PSX query-system evaluation.")
    ap.add_argument("--records", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--today", required=True, help="YYYY-MM-DD, anchors relative dates")
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()
    records = load_corpus(args.records)
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    out = run_eval(gold, records, today=args.today, trials=args.trials)
    summary = {k: out[k] for k in ["translation_exact", "answer_correct", "answer_grounded"]}
    print(json.dumps({"summary": summary, "schema_coverage": out["schema_coverage"],
                      "trials": out["trials"]}, indent=2))

if __name__ == "__main__":
    main()
