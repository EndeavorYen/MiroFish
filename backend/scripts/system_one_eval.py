"""Evaluate System One readout on tests/fixtures/system_one_eval.jsonl.

Reports per question type: top-1 accuracy, ECE before/after temperature
calibration (2-fold cross-validated), mean latency, and the latency of a
second question on a state whose prefix is already cached. Optionally writes
the temperatures fitted on the full set to app/system_one/calibration.json.

Usage:
    uv run python scripts/system_one_eval.py \
        --base-url http://127.0.0.1:8000/v1 --model qwen3.5-4b \
        [--write-calibration] [--out report.json]
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
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.system_one.backends import LocalReadoutBackend  # noqa: E402
from app.system_one.calibration import (  # noqa: E402
    expected_calibration_error,
    fit_temperature,
    top1,
)
from app.system_one.client import CALIBRATION_PATH  # noqa: E402
from app.system_one.models import (  # noqa: E402
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
)

DEFAULT_EVAL = BACKEND_DIR / "tests" / "fixtures" / "system_one_eval.jsonl"
NEWS_SEED = BACKEND_DIR / "tests" / "fixtures" / "golden_scenario" / "news_seed.txt"
G2_THRESHOLD = 0.70


def question_for(row: dict[str, Any]):
    if row["type"] == "choice":
        return ChoiceQuestion(instructions=row["instructions"], criteria=row["criteria"])
    if row["type"] == "score":
        return ScoreQuestion(instructions=row["instructions"], criteria=row["criteria"])
    return NoulQuestion(instructions=row["instructions"])


def distribution(row: dict[str, Any], answer) -> tuple[list[float], int]:
    """Return (probabilities in label order, gold index)."""

    if row["type"] == "choice":
        keys = list(row["criteria"])
        return [answer.probabilities[k] for k in keys], keys.index(row["label"])
    if row["type"] == "score":
        return [answer.probabilities[c] for c in row["criteria"]], int(row["label"])
    return [answer.noul, 1.0 - answer.noul], 0 if row["label"] else 1


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    dists = [r["probs"] for r in records]
    labels = [r["gold"] for r in records]
    conf, ok = top1(dists, labels)
    # 2-fold CV so "after" ECE is not measured on the data T was fitted on.
    folds = [records[0::2], records[1::2]]
    cv_conf, cv_ok, fold_temps = [], [], []
    for held, fit in ((folds[0], folds[1]), (folds[1], folds[0])):
        t = fit_temperature([r["probs"] for r in fit], [r["gold"] for r in fit])
        fold_temps.append(t)
        c, o = top1([r["probs"] for r in held], [r["gold"] for r in held], t)
        cv_conf += c
        cv_ok += o
    t_all = fit_temperature(dists, labels)
    conf_all, ok_all = top1(dists, labels, t_all)
    latencies = [r["latency_ms"] for r in records]
    return {
        "n": len(records),
        "top1_accuracy": round(sum(ok) / len(ok), 4),
        "ece_before": round(expected_calibration_error(conf, ok), 4),
        "ece_after_cv": round(expected_calibration_error(cv_conf, cv_ok), 4),
        "ece_after_in_sample": round(expected_calibration_error(conf_all, ok_all), 4),
        "temperature": t_all,
        "fold_temperatures": fold_temps,
        "mean_coverage": round(statistics.mean(r["coverage"] for r in records), 4),
        "latency_ms_mean": round(statistics.mean(latencies), 1),
        "latency_ms_p50": round(statistics.median(latencies), 1),
    }


def prefix_cache_probe(backend: LocalReadoutBackend, rows: list[dict[str, Any]], n: int) -> dict:
    """Ask two different questions on each state; the second hits the cache.

    The state is simulation-sized (golden news seed + one post), because a
    ~50-token eval state leaves nothing worth caching.
    """

    first, second = [], []
    context = NEWS_SEED.read_text(encoding="utf-8")
    stance = next(r for r in rows if r["category"] == "post_stance")
    probe_q = ChoiceQuestion(instructions=stance["instructions"], criteria=stance["criteria"])
    for i, row in enumerate(r for r in rows if r["category"] == "post_sentiment"):
        if i >= n:
            break
        # A per-run nonce keeps the first question cold even on a warm server.
        state = f"[probe {time.time_ns()}]\n{context}\n\n{row['state']}"
        for bucket, question in ((first, question_for(row)), (second, probe_q)):
            started = time.perf_counter()
            backend._ask_one(state, question)
            bucket.append((time.perf_counter() - started) * 1000)
    return {
        "n": len(first),
        "state_chars": len(context),
        "first_question_ms_mean": round(statistics.mean(first), 1),
        "second_question_ms_mean": round(statistics.mean(second), 1),
        "first_question_ms_p50": round(statistics.median(first), 1),
        "second_question_ms_p50": round(statistics.median(second), 1),
        # Medians: a retried loopback reset adds a 250 ms sleep to one sample.
        "reduction_p50": round(1 - statistics.median(second) / statistics.median(first), 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen3.5-4b")
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--prompt-format", default="chatml", choices=["chatml", "plain"])
    parser.add_argument("--probe", type=int, default=30, help="states for the prefix-cache probe")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--write-calibration", action="store_true")
    args = parser.parse_args(argv)

    all_rows = [
        json.loads(line) for line in args.eval.read_text(encoding="utf-8").splitlines() if line
    ]
    rows = all_rows[: args.limit] if args.limit else all_rows
    # Raw readout: temperature 1 everywhere.
    backend = LocalReadoutBackend(
        base_url=args.base_url,
        model=args.model,
        top_k=args.top_k,
        prompt_format=args.prompt_format,
    )

    by_type: dict[str, list[dict[str, Any]]] = {}
    misses: list[dict[str, Any]] = []
    for row in rows:
        question = question_for(row)
        started = time.perf_counter()
        answer, _, _ = backend._ask_one(row["state"], question)
        latency_ms = (time.perf_counter() - started) * 1000
        probs, gold = distribution(row, answer)
        record = {
            "id": row["id"],
            "probs": probs,
            "gold": gold,
            "coverage": answer.coverage or 0.0,
            "latency_ms": latency_ms,
        }
        by_type.setdefault(row["type"], []).append(record)
        if max(range(len(probs)), key=lambda i: probs[i]) != gold:
            misses.append({"id": row["id"], "gold": gold, "probs": [round(p, 3) for p in probs]})

    report: dict[str, Any] = {
        "model": args.model,
        "base_url": args.base_url,
        "prompt_format": args.prompt_format,
        "top_k": args.top_k,
        "eval_items": len(rows),
        "by_type": {qtype: summarize(records) for qtype, records in sorted(by_type.items())},
        "prefix_cache": prefix_cache_probe(backend, all_rows, args.probe),
        "misses": misses,
    }
    choice_acc = report["by_type"].get("choice", {}).get("top1_accuracy")
    report["gate_g2"] = {
        "threshold": G2_THRESHOLD,
        "choice_top1": choice_acc,
        "passed": choice_acc is not None and choice_acc >= G2_THRESHOLD,
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    if args.write_calibration:
        # Keep a fitted T only when it lowers held-out ECE; otherwise 1.0.
        temps = {
            qtype: summary["temperature"]
            if summary["ece_after_cv"] < summary["ece_before"]
            else 1.0
            for qtype, summary in report["by_type"].items()
        }
        CALIBRATION_PATH.write_text(
            json.dumps(
                {
                    "temperatures": temps,
                    "fitted_on": args.eval.name,
                    "model": args.model,
                    "prompt_format": args.prompt_format,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
