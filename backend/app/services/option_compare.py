"""Compare options on one scenario by the reaction they get (#56, #66).

Each option is the scenario with the option added to the document and the
requirement; every option shares the baseline's graph and runs on the same
seeds, and its posts are scored on the same question (the base scenario's
event), so the model's own biases largely cancel between options and the
per-seed differences can be paired. The first option is the baseline.

Used by ``scripts/compare_options.py`` and by a run with ``options``.
"""

from __future__ import annotations

import re
import statistics
from typing import Any

MIN_OPTIONS, MAX_OPTIONS = 2, 4
MAX_TEXT = 300  # an option is announced in the requirement: keep the prompts small
MAX_RUNS = 16  # options x seeds
# All seeds moving the same way by chance: 1 in 2 with two seeds, 1 in 4 with
# three. Fewer than three seeds never tell options apart.
MIN_SEEDS_TO_DISTINGUISH = 3
MAX_NAME = 40


def validate(options: Any) -> list[dict[str, str]]:
    """2-4 options with a unique, non-empty name and a text; ``ValueError``
    says what is wrong. The first is the baseline."""

    if not isinstance(options, list) or not MIN_OPTIONS <= len(options) <= MAX_OPTIONS:
        raise ValueError(f"options must be a list of {MIN_OPTIONS} to {MAX_OPTIONS}; the first is the baseline")
    out = [validate_one(option) for option in options]
    if len({o["name"] for o in out}) != len(out):
        raise ValueError("option names must be unique")
    return out


def validate_one(option: Any) -> dict[str, str]:
    """One option: a non-empty name (at most 40 characters) and a text."""

    if not isinstance(option, dict):
        raise ValueError("every option is an object with a name and a text")
    name, text = option.get("name"), option.get("text")
    if not isinstance(name, str) or not isinstance(text, str) or not name.strip() or not text.strip():
        raise ValueError("every option needs a name and a text")
    if len(name.strip()) > MAX_NAME:
        raise ValueError(f"an option name has at most {MAX_NAME} characters")
    if len(text.strip()) > MAX_TEXT:
        raise ValueError(f"an option text has at most {MAX_TEXT} characters")
    return {"name": name.strip(), "text": text.strip()}


def with_option(requirement: str, document: str, option: dict[str, str]) -> tuple[str, str]:
    """The requirement and the document with the option announced first in
    both: the local prep reads the first 300 characters of the requirement and
    opens the simulation with the document's first sentences, so an option
    added at the end would not reach the agents."""

    from ..simulation_policy.tiers import detect_content_lang

    english = detect_content_lang(requirement) == "en"
    text = option["text"].rstrip()
    if text[-1] not in ".。!！?？":
        text += "." if english else "。"
    addition = f"Announced plan: {text}" if english else f"公布的方案：{text}"
    return f"{addition}\n{requirement}", f"{addition}\n\n{document.strip()}\n"


def record(config_path: str, base_requirement: str, option: dict[str, str]) -> None:
    """Note in a prepared simulation's config which option it was prepared
    with, and the requirement without it: the report scores its posts on the
    base requirement's question, like the comparison does."""

    import json

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    config["option"] = option
    config["base_requirement"] = base_requirement
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def run_measures(scan: dict[str, Any]) -> dict[str, float]:
    return {"tendency": scan["tendency"]["value"], "oppose_share": scan["post_shares"]["oppose"]}


def paired(base: dict[int, dict[str, float]], other: dict[int, dict[str, float]], key: str) -> dict[str, Any]:
    """Per-seed differences other - base on the seeds both have."""

    diffs = [other[s][key] - base[s][key] for s in sorted(set(base) & set(other))]
    if not diffs:
        return {"mean": None, "sd": None, "seeds": 0, "same_sign": 0, "distinct": False}
    positive = sum(1 for d in diffs if d > 0)
    negative = sum(1 for d in diffs if d < 0)
    same = max(positive, negative)
    return {
        "mean": round(statistics.mean(diffs), 4),
        "sd": round(statistics.pstdev(diffs), 4),
        "range": [round(min(diffs), 4), round(max(diffs), 4)],
        "seeds": len(diffs),
        "same_sign": same,
        # Every seed moves the same way, on at least 3 seeds; with 3 that is
        # still a 1-in-4 chance under no effect, so use 5 seeds before acting
        # on a small difference.
        "distinct": len(diffs) >= MIN_SEEDS_TO_DISTINGUISH and same == len(diffs),
    }


def compare(results: dict[str, dict[int, dict[str, Any]]], names: list[str]) -> list[dict[str, Any]]:
    """Rows per option from ``results[name][seed] = scan``; options without
    any scored run are left out, and without the baseline (the first name)
    nothing can be compared."""

    if not names or not results.get(names[0]):
        return []
    names = [n for n in names if results.get(n)]
    measures = {n: {s: run_measures(scan) for s, scan in results[n].items()} for n in names}
    base = names[0]
    rows = []
    for name in names:
        runs = measures[name]
        row: dict[str, Any] = {
            "option": name,
            "seeds": sorted(runs),
            "tendency": round(statistics.mean(r["tendency"] for r in runs.values()), 4),
            "oppose_share": round(statistics.mean(r["oppose_share"] for r in runs.values()), 4),
            "most_opposed_posts": sorted(
                (p for scan in results[name].values() for p in scan["most_opposed_posts"]),
                key=lambda p: (p["stance"], p["text"]),
            )[:3],
        }
        if name != base:
            row["vs_baseline"] = {
                "tendency": paired(measures[base], runs, "tendency"),
                "oppose_share": paired(measures[base], runs, "oppose_share"),
            }
        rows.append(row)
    return rows
