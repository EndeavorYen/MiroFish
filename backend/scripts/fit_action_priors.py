"""Fit System One action priors from calibration runs (#26).

Inputs are ``golden_pipeline.py simulate`` runs on calibration seeds that the
A/B evaluation does not use:

* ``--a-runs``: LLM-decision runs; their actions give the target mix;
* ``--b-runs``: System One runs made without priors; their decisions.jsonl
  holds the raw readout at every tree node an agent reached.

For each tree node, the weights ``w`` of the non-exit options solve
``mean_r(normalise(r * w))_i = M * f_i`` over the recorded readouts ``r``,
where ``f`` is the A action mix over that node's options (add-0.5 smoothed)
and ``M`` the readouts' original non-exit mass, so priors change *which*
action is taken, not how often agents act. Exit options (none/other) keep
weight 1: LLM agents never log DO_NOTHING, so there is no target for them.

Usage:
    uv run python scripts/fit_action_priors.py --a-runs <A_seed11> ... \\
        --b-runs <B_seed11> ... --out app/simulation_policy/action_priors.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.simulation_policy.taxonomy import EXIT_KEYS, Taxonomy, load_taxonomy  # noqa: E402

PLATFORMS = ("twitter", "reddit")
SMOOTHING = 0.5
ITERATIONS = 300
W_MIN, W_MAX = 1e-3, 1e3


def node_options(taxonomy: Taxonomy) -> dict[str, list[str]]:
    """node key ("" = root, "engage" ...) -> option keys."""

    result: dict[str, list[str]] = {}

    def walk(node: dict[str, Any], path: tuple[str, ...]) -> None:
        result["/".join(path)] = list(node["question"]["criteria"])
        for key, child in (node.get("children") or {}).items():
            walk(child, path + (key,))

    walk(taxonomy.tree, ())
    return result


def action_paths(taxonomy: Taxonomy) -> dict[str, tuple[str, ...]]:
    """OASIS action -> its (first) taxonomy path, exits excluded."""

    paths: dict[str, tuple[str, ...]] = {}
    for path, leaf in sorted(taxonomy.leaves.items()):
        if path[-1] in EXIT_KEYS:
            continue
        paths.setdefault(leaf.action, path)
    return paths


def target_counts(actions: Counter, taxonomy: Taxonomy) -> dict[str, Counter]:
    """Per node, how often A's actions went through each option."""

    paths = action_paths(taxonomy)
    counts: dict[str, Counter] = defaultdict(Counter)
    for action, n in actions.items():
        path = paths.get(action.upper())
        if path is None:
            continue
        for depth in range(len(path)):
            counts["/".join(path[:depth])][path[depth]] += n
    return counts


def readouts_by_node(decisions: list[dict[str, Any]]) -> dict[str, list[dict[str, float]]]:
    result: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in decisions:
        probs = row.get("raw_probs") or row.get("probs") or []
        for depth, step in enumerate(probs):
            result["/".join(row["path"][:depth])].append(step)
    return result


def _normalise(p: dict[str, float], w: dict[str, float]) -> dict[str, float]:
    weighted = {k: v * w.get(k, 1.0) for k, v in p.items()}
    total = sum(weighted.values()) or 1.0
    return {k: v / total for k, v in weighted.items()}


def fit_node(options: list[str], counts: Counter, readouts: list[dict[str, float]]) -> dict[str, float]:
    """Weights for one node (exit options fixed at 1.0)."""

    active = [o for o in options if o not in EXIT_KEYS]
    if not active or not sum(counts.values()):
        return {}
    if not readouts:
        readouts = [{o: 1.0 / len(options) for o in options}]
    readouts = [{o: float(r.get(o, 0.0)) for o in options} for r in readouts]
    smoothed = {o: counts.get(o, 0) + SMOOTHING for o in active}
    total = sum(smoothed.values())
    mass = sum(sum(r[o] for o in active) for r in readouts) / len(readouts)
    target = {o: mass * smoothed[o] / total for o in active}
    w = {o: 1.0 for o in options}
    for _ in range(ITERATIONS):
        adjusted = defaultdict(float)
        for r in readouts:
            for k, v in _normalise(r, w).items():
                adjusted[k] += v / len(readouts)
        for o in active:
            if adjusted[o] > 0:
                w[o] = min(W_MAX, max(W_MIN, w[o] * target[o] / adjusted[o]))
    return {o: round(w[o], 4) for o in options}


def fit_platform(actions: Counter, decisions: list[dict[str, Any]], taxonomy: Taxonomy) -> dict[str, dict[str, float]]:
    counts = target_counts(actions, taxonomy)
    readouts = readouts_by_node(decisions)
    priors = {}
    for node, options in node_options(taxonomy).items():
        weights = fit_node(options, counts.get(node, Counter()), readouts.get(node, []))
        if weights:
            priors[node] = weights
    return priors


def _run_actions(run: Path) -> dict[str, Counter]:
    """platform -> Counter(action) from a run's sim/<platform>/actions.jsonl."""

    result: dict[str, Counter] = {p: Counter() for p in PLATFORMS}
    for platform in PLATFORMS:
        path = run / "sim" / platform / "actions.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "event_type" in row or not row.get("action_type"):
                continue
            result[platform][str(row["action_type"]).upper()] += 1
    return result


def _run_decisions(run: Path) -> list[dict[str, Any]]:
    path = run / "sim" / "decisions.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a-runs", nargs="+", required=True, type=Path)
    parser.add_argument("--b-runs", nargs="+", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    actions = {p: Counter() for p in PLATFORMS}
    for run in args.a_runs:
        for platform, counts in _run_actions(run).items():
            actions[platform].update(counts)
    decisions = [row for run in args.b_runs for row in _run_decisions(run)]
    if any(row.get("raw_probs") for row in decisions):
        print("note: B runs already used priors; fitting on their raw readouts", file=sys.stderr)

    priors = {
        platform: fit_platform(actions[platform], [d for d in decisions if d.get("platform") == platform],
                               load_taxonomy(platform))
        for platform in PLATFORMS
    }
    out = {
        "fitted_on": {
            "a_runs": [run.name for run in args.a_runs],
            "b_runs": [run.name for run in args.b_runs],
            "a_actions": {p: dict(c) for p, c in actions.items()},
            "b_decisions": len(decisions),
        },
        "method": "per node: mean(normalise(readout*w)) = M*f over recorded readouts; exits fixed at 1",
        "priors": priors,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(priors, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
