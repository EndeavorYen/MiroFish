"""Fit System One action priors from calibration runs (#26).

Inputs are ``golden_pipeline.py simulate`` runs on calibration seeds that the
A/B evaluation does not use:

* ``--a-runs``: LLM-decision runs; their actions give the target mix;
* ``--b-runs``: System One runs made without priors; their decisions.jsonl
  holds the raw readout at every tree node an agent reached.

For each tree node, the weights ``w`` solve ``mean_r(normalise(r * w)) = f``
over the recorded readouts ``r``, where ``f`` is the A action mix over that
node's options (add-0.5 smoothed). Round 0 (the scripted initial posts) is
not an agent decision and is left out.

LLM agents never log DO_NOTHING, so the exits' targets come from activity:
an activation is one LLM call; the share of calls after which the agent
took no action is the root exit's target, and exits below the root target
zero (an agent that chose "engage" engages). LLM agents also take ~1.6
actions per activation; ``extra_action_rate`` = 1 - activations/actions is
the chance of one more action after each action (see
``SystemOnePolicy.decide_round``).

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


def fit_node(
    options: list[str], counts: Counter, readouts: list[dict[str, float]], *, fit_exits: bool = False
) -> dict[str, float]:
    """Weights for one node. Without ``fit_exits`` the exits keep weight 1 and
    the non-exit mass of the readouts is preserved; with it every option,
    exits included, is fitted to ``counts``."""

    active = list(options) if fit_exits else [o for o in options if o not in EXIT_KEYS]
    if not active or not sum(counts.get(o, 0) for o in options if o not in EXIT_KEYS):
        return {}
    if not readouts:
        readouts = [{o: 1.0 / len(options) for o in options}]
    readouts = [{o: float(r.get(o, 0.0)) for o in options} for r in readouts]
    smoothed = {o: counts.get(o, 0) + SMOOTHING for o in active}
    total = sum(smoothed.values())
    mass = 1.0 if fit_exits else sum(sum(r[o] for o in active) for r in readouts) / len(readouts)
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


def fit_platform(
    actions: Counter,
    decisions: list[dict[str, Any]],
    taxonomy: Taxonomy,
    no_action_share: float | None = None,
) -> dict[str, dict[str, float]]:
    """Per-node weights. With ``no_action_share`` the exits are fitted too:
    the root exit to that share of activations, exits below it to zero."""

    counts = target_counts(actions, taxonomy)
    if no_action_share is not None:
        first = sum(n for o, n in counts.get("", Counter()).items())
        share = min(max(no_action_share, 0.0), 0.95)
        for exit_key in EXIT_KEYS:
            if exit_key in node_options(taxonomy)[""]:
                counts.setdefault("", Counter())[exit_key] = share / (1 - share) * first
                break
    readouts = readouts_by_node(decisions)
    priors = {}
    for node, options in node_options(taxonomy).items():
        weights = fit_node(
            options, counts.get(node, Counter()), readouts.get(node, []),
            fit_exits=no_action_share is not None,
        )
        if weights:
            priors[node] = weights
    return priors


def activity(run: Path) -> dict[str, Any]:
    """LLM calls, and per platform: agent-rounds with an action and actions (round > 0)."""

    usage = run / "metrics" / "llm_usage.jsonl"
    calls = 0
    if usage.exists():
        calls = sum(
            1 for line in usage.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("stage") == "simulation"
        )
    acted: dict[str, set] = {p: set() for p in PLATFORMS}
    actions: dict[str, int] = {p: 0 for p in PLATFORMS}
    for platform in PLATFORMS:
        path = run / "sim" / platform / "actions.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "event_type" in row or not row.get("action_type") or int(row.get("round", 0)) == 0:
                continue
            acted[platform].add((row.get("round"), row.get("agent_id")))
            actions[platform] += 1
    return {"calls": calls, "acted": {p: len(v) for p, v in acted.items()}, "actions": actions}


def _run_actions(run: Path) -> dict[str, Counter]:
    """platform -> Counter(action) from a run's sim/<platform>/actions.jsonl."""

    result: dict[str, Counter] = {p: Counter() for p in PLATFORMS}
    for platform in PLATFORMS:
        path = run / "sim" / platform / "actions.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            # Round 0 is the scripted initial posts, not agent decisions.
            if "event_type" in row or not row.get("action_type") or int(row.get("round", 0)) == 0:
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

    runs_activity = [activity(run) for run in args.a_runs]
    calls = sum(a["calls"] for a in runs_activity)
    acted = {p: sum(a["acted"][p] for a in runs_activity) for p in PLATFORMS}
    acted_actions = {p: sum(a["actions"][p] for a in runs_activity) for p in PLATFORMS}
    no_action_share = max(0.0, 1 - sum(acted.values()) / calls) if calls else 0.0
    extra_rate = {
        p: round(min(0.9, max(0.0, 1 - acted[p] / acted_actions[p])), 4) if acted_actions[p] else 0.0
        for p in PLATFORMS
    }
    priors = {
        platform: fit_platform(actions[platform], [d for d in decisions if d.get("platform") == platform],
                               load_taxonomy(platform), no_action_share)
        for platform in PLATFORMS
    }
    out = {
        "fitted_on": {
            "a_runs": [run.name for run in args.a_runs],
            "b_runs": [run.name for run in args.b_runs],
            "a_actions": {p: dict(c) for p, c in actions.items()},
            "b_decisions": len(decisions),
            "a_llm_calls": calls,
            "a_acted_agent_rounds": acted,
            "a_no_action_share": round(no_action_share, 4),
        },
        "method": "per node: mean(normalise(readout*w)) = f over recorded readouts; root exit = A no-action share",
        "priors": priors,
        "extra_action_rate": extra_rate,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(priors, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
