"""Readout accuracy, bias and temperatures for one pool model (#48).

Runs the System One eval set (tests/fixtures/system_one_eval.jsonl) against
one OpenAI-compatible endpoint and reports, besides per-type accuracy:

* positivity bias: mean predicted minus mean gold position on
  post_sentiment (negative, neutral, positive) and post_stance (oppose,
  neutral, support; "other" left out), in levels; > 0 reads text as more
  positive than it is (the #41 bias);
* engage bias: mean p(yes) minus the gold rate on will_engage (the #26 lean
  toward acting).

``--write`` stores the temperatures under ``models.<model>`` in
app/system_one/calibration.json; the top-level entry is left alone.

Usage:
    uv run python scripts/calibrate_model.py --base-url http://127.0.0.1:8002/v1 \\
        --model gemma-3-4b --prompt-format plain --out gemma.json [--write]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "scripts"))
import system_one_eval as ev  # noqa: E402
from app.system_one.backends import LocalReadoutBackend  # noqa: E402
from app.system_one.client import CALIBRATION_PATH  # noqa: E402

# Label order from negative to positive; labels not listed are left out.
POLARITY = {
    "post_sentiment": ["negative", "neutral", "positive"],
    "post_stance": ["oppose", "neutral", "support"],
}


def polarity_bias(records: list[dict[str, Any]]) -> float | None:
    """Mean (expected predicted position - gold position) over polar items."""

    gaps = []
    for record in records:
        order = POLARITY.get(record["category"])
        if not order or record["label"] not in order:
            continue
        probs = {key: p for key, p in zip(record["keys"], record["probs"]) if key in order}
        total = sum(probs.values())
        if total <= 0:
            continue
        expected = sum(order.index(key) * p for key, p in probs.items()) / total
        gaps.append(expected - order.index(record["label"]))
    return round(statistics.mean(gaps), 4) if gaps else None


def engage_bias(records: list[dict[str, Any]]) -> float | None:
    rows = [r for r in records if r["category"] == "will_engage"]
    if not rows:
        return None
    return round(statistics.mean(r["probs"][0] for r in rows) - statistics.mean(1.0 if r["label"] else 0.0 for r in rows), 4)


def parse_with(spec: str) -> tuple[str, str, str]:
    """``base_url,model[,prompt_format]`` of one more ensemble member."""

    parts = [p.strip() for p in spec.split(",")]
    if len(parts) not in (2, 3) or not all(parts):
        raise ValueError(f"--with wants base_url,model[,prompt_format], got {spec!r}")
    return parts[0], parts[1], parts[2] if len(parts) == 3 else "chatml"


def ask_all(backends: list, state: str, row: dict[str, Any]):
    """One answer per backend, averaged the way the readout ensemble does (#48)."""

    from app.model_pool import combine

    question = ev.question_for(row)
    return combine(question, [backend._ask_one(state, question)[0] for backend in backends])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-format", default="chatml", choices=["chatml", "plain"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--eval", type=Path, default=ev.DEFAULT_EVAL)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--with", dest="members", action="append", default=[], metavar="URL,MODEL[,FORMAT]",
                        help="another model; the report is then for the averaged ensemble (no --write)")
    args = parser.parse_args(argv)
    if args.members and args.write:
        parser.error("--write stores one model's temperatures; drop --with")

    rows = [json.loads(line) for line in args.eval.read_text(encoding="utf-8").splitlines() if line]
    backends = [
        LocalReadoutBackend(base_url=url, model=model, top_k=args.top_k, prompt_format=fmt)
        for url, model, fmt in [(args.base_url, args.model, args.prompt_format), *map(parse_with, args.members)]
    ]
    by_type: dict[str, list[dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    for row in rows:
        started = time.perf_counter()
        answer = ask_all(backends, row["state"], row)
        latency_ms = (time.perf_counter() - started) * 1000
        probs, gold = ev.distribution(row, answer)
        record = {
            "id": row["id"], "category": row["category"], "label": row["label"],
            "keys": list(row["criteria"]) if row["type"] == "choice" else [],
            "probs": probs, "gold": gold, "coverage": answer.coverage or 0.0, "latency_ms": latency_ms,
        }
        records.append(record)
        by_type.setdefault(row["type"], []).append(record)
    by_category: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_category.setdefault(record["category"], []).append(record)
    report = {
        "model": args.model,
        "ensemble_with": args.members,
        "base_url": args.base_url,
        "prompt_format": args.prompt_format,
        "eval_items": len(rows),
        "by_type": {t: ev.summarize(r) for t, r in sorted(by_type.items())},
        "accuracy_by_category": {
            c: round(sum(max(range(len(r["probs"])), key=r["probs"].__getitem__) == r["gold"] for r in rs) / len(rs), 4)
            for c, rs in sorted(by_category.items())
        },
        "latency_ms_mean": round(statistics.mean(r["latency_ms"] for r in records), 1),
        "positivity_bias": polarity_bias(records),
        "engage_bias": engage_bias(records),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    if args.write:
        temps = {
            t: s["temperature"] if s["ece_after_cv"] < s["ece_before"] else 1.0
            for t, s in report["by_type"].items()
        }
        data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8")) if CALIBRATION_PATH.exists() else {}
        data.setdefault("models", {})[args.model] = {
            "temperatures": temps, "prompt_format": args.prompt_format, "fitted_on": args.eval.name,
        }
        CALIBRATION_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
