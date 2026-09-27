"""Build locales/<lang>_stance_bank.json from calibration LLM runs (#45).

Each ``--dirs`` entry is a calibration scenario dir made by ``ab_suite.py
--manifest tests/fixtures/calibration/calibration.json``: ``A/`` (LLM prep)
and ``runs/A_seed<N>`` (LLM-decision runs). Posts are scored with the
scenario's own stance question (the same readout ab_eval uses), kept when one
of the five levels has probability >= ``--min-prob``, and de-identified: the
first seed entity becomes ``{entity}``; a post naming a second one is dropped.

Only calibration seeds are read; evaluation seeds 1-5 are refused.

Usage:
    uv run python scripts/build_stance_bank.py --lang zh --dirs <cal>/<scenario> ... \\
        --out ../locales/zh_stance_bank.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "scripts"))
import ab_eval  # noqa: E402
import golden_pipeline as gp  # noqa: E402
from app.simulation_policy.tiers import EVAL_SEEDS, LEVELS  # noqa: E402

BLOCKED = ("@", "http", "{", "}")
_HASHTAG_RE = re.compile(r"#[^#\s]+#?")


def deidentify(text: str, names: list[str]) -> str | None:
    """``text`` without hashtags and with its one named entity as ``{entity}``;
    None when it names more than one, or carries mentions, links or braces."""

    if any(mark in text for mark in BLOCKED):
        return None
    text = _HASHTAG_RE.sub("", text)
    found = [name for name in sorted(set(names), key=len, reverse=True) if name and name in text]
    kept: list[str] = []
    for name in found:
        if not any(name in longer for longer in kept):
            kept.append(name)
    if len(kept) > 1:
        return None
    if kept:
        text = text.replace(kept[0], "{entity}")
        if text.count("{entity}") > 1:
            return None
    return " ".join(text.split())


def usable_length(text: str, lang: str) -> bool:
    if lang == "en":
        return 5 <= len(text.split()) <= 60
    return 8 <= len(text) <= 120


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--lang", choices=["zh", "en"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--min-prob", type=float, default=0.6)
    parser.add_argument("--per-level", type=int, default=40)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    overlap = sorted(set(args.seeds) & EVAL_SEEDS)
    if overlap:
        raise SystemExit(f"the bank must not use evaluation seeds {overlap}")

    gp.force_local_config(args.out.parent)
    from app.system_one.client import get_system_one_client

    client = get_system_one_client()
    per_scenario: list[dict[str, list[str]]] = []
    sources: list[str] = []
    for scenario in args.dirs:
        work = scenario / "A"
        prepared = json.loads((work / "prepared.json").read_text(encoding="utf-8"))
        question = ab_eval.stance_question_for(prepared)
        rows = [row for seed in args.seeds for row in ab_eval.post_rows(scenario / "runs" / f"A_seed{seed}")]
        names = gp.entity_names(work, prepared) + [row["agent"] for row in rows]
        cache: dict[str, dict[str, Any]] = {}
        ab_eval.score_posts(client, [row["text"] for row in rows], cache, question=question)
        levels: dict[str, list[str]] = {level: [] for level in LEVELS}
        for row in rows:
            probs = cache[row["text"]]["levels"]
            best = max(range(len(probs)), key=probs.__getitem__)
            if probs[best] < args.min_prob:
                continue
            line = deidentify(row["text"], names)
            if line and usable_length(line, args.lang) and line not in levels[LEVELS[best]]:
                levels[LEVELS[best]].append(line)
        per_scenario.append(levels)
        sources.append(scenario.name)
    # Round-robin across scenarios so no single event dominates a level.
    bank: dict[str, list[str]] = {level: [] for level in LEVELS}
    for level in LEVELS:
        queues = [list(levels[level]) for levels in per_scenario]
        while len(bank[level]) < args.per_level and any(queues):
            for queue in queues:
                if queue and len(bank[level]) < args.per_level:
                    bank[level].append(queue.pop(0))
    out = {"lang": args.lang, "seeds": args.seeds, "sources": sources, "min_prob": args.min_prob, "levels": bank}
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print({level: len(lines) for level, lines in bank.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
