"""A/B fidelity evaluation on the golden scenario (#13).

Group A (baseline): LLM decisions on the LLM-prepared scenario.
Group B (local):    System One decisions + tiered content on the
                    template/structured-prepared scenario.
Both run on the local graph and the local model server (no external API).

For each group and seed, ``golden_pipeline.py simulate`` is run (or an
existing run is reused), then this script computes:

* decode tokens per round, round latency, VRAM peak;
* Jensen-Shannon divergence of action-type distributions: A seed vs A seed
  (the noise floor) and B vs A;
* stance (#27, G4 v2 in #44): every post is scored with that scenario's
  ``stance_question.txt`` (or ``--stance-question``; otherwise
  ``STANCE_QUESTION``). Gated with a seed-level bootstrap (1,000 reps,
  RNG seed 0): action-mix JS and stance-level JS fail when the one-sided
  p-value of B-vs-A against A-within is < 0.05; per-persona correlation
  passes when its 95% CI lower bound is at least the median A-pair
  correlation minus delta (default 0.15), with at least 12 shared personas
  and 80% of A's. Fewer than 12 shared personas is undecidable, not a
  failure. Reported only: the per-round curve and a 4-round windowed curve;
* distinct-2 and seed-entity mention rate of posts;
* extraction recall from #7 (passed in).

It writes ``ab_report.json`` and ``ab_report.md`` with the G4 gate result and
a recommendation (switch defaults, switch conditionally, or keep llm).

Runs are reused when their summary.json exists, so each group can first be
simulated under its own model-server settings with ``--simulate-only A|B``
(OASIS LLM agents in A need about 32K context per slot).

Usage:
    uv run python scripts/ab_eval.py --work-a <llm prepared> --work-b <template prepared> \
        --out <dir> --seeds 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import statistics
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR / "scripts"))
import golden_pipeline as gp  # noqa: E402  (captures the pre-.env environment)

STANCE_LEVELS = ["強烈反對", "反對", "中立", "支持", "強烈支持"]
STANCE_QUESTION = "這則貼文對「東海市無人駕駛空中計程車試點」的立場是什麼？"
VRAM_BUDGET_MIB = 10 * 1024


# ------------------------------------------------------------------ maths


def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon divergence (base 2, 0..1) of two count dicts."""

    keys = sorted(set(p) | set(q))
    ps = sum(p.values()) or 1.0
    qs = sum(q.values()) or 1.0
    pv = [p.get(k, 0) / ps for k in keys]
    qv = [q.get(k, 0) / qs for k in keys]
    mv = [(a + b) / 2 for a, b in zip(pv, qv)]

    def kl(x, y):
        return sum(a * math.log2(a / b) for a, b in zip(x, y) if a > 0)

    return 0.5 * kl(pv, mv) + 0.5 * kl(qv, mv)


def pearson(x: list[float], y: list[float]) -> float | None:
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in pairs) / (sx * sy)


# ------------------------------------------------------------------- runs


def run_simulation(work: Path, out: Path, backend: str, seed: int, rounds: int, content: str | None) -> dict:
    summary = out / "summary.json"
    if summary.exists():
        data = json.loads(summary.read_text(encoding="utf-8"))
        check_reused(out, data, backend, seed, rounds)
        return data
    command = [
        sys.executable, str(BACKEND_DIR / "scripts" / "golden_pipeline.py"), "simulate",
        "--work", str(work), "--out", str(out), "--decision-backend", backend,
        "--seed", str(seed), "--max-rounds", str(rounds),
    ]
    if content:
        command += ["--content-mode", content]
    subprocess.run(command, cwd=str(BACKEND_DIR), check=True, stdout=subprocess.DEVNULL)
    return json.loads(summary.read_text(encoding="utf-8"))


def simulate_many(tasks: list[tuple], jobs: int = 1) -> dict[tuple, dict[str, Any]]:
    """``run_simulation(*task)`` for every task, ``jobs`` at a time.

    Runs are independent, and one System One run keeps the model server's
    GPU mostly idle (7-29% utilisation; its rounds wait on sequential
    readouts), so several runs side by side finish sooner. Tasks may come
    from several scenarios. Round latency is then measured under contention;
    decode counts are not affected."""

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        return dict(zip(tasks, pool.map(lambda task: run_simulation(*task), tasks)))


def run_all(
    groups: dict[str, dict[str, Any]], out: Path, seeds: list[int], rounds: int, jobs: int = 1
) -> dict[str, dict[int, dict[str, Any]]]:
    """Every (group, seed) run of one scenario, ``jobs`` at a time."""

    keys = [(name, seed) for name in groups for seed in seeds]
    tasks = [
        (groups[n]["work"], out / f"{n}_seed{s}", groups[n]["backend"], s, rounds, groups[n]["content"])
        for n, s in keys
    ]
    summaries = simulate_many(tasks, jobs)
    runs: dict[str, dict[int, dict[str, Any]]] = {name: {} for name in groups}
    for (name, seed), task in zip(keys, tasks):
        runs[name][seed] = {"dir": task[1], "summary": summaries[task]}
    return runs


def llm_errors(run: Path) -> int:
    """Model-server errors (context overflow, 5xx) the simulation logged.

    A run with errors lost agent turns, so its numbers are not comparable.
    """

    path = run / "simulation.log"
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8", errors="replace")
    return sum(1 for line in text.splitlines() if re.search(r"Error code: [45]\d\d", line))


def check_reused(out: Path, data: dict, backend: str, seed: int, rounds: int) -> None:
    """A reused run must be a clean run of the same settings."""

    expected = {"exit_code": 0, "decision_backend": backend, "seed": seed, "max_rounds": rounds}
    wrong = {k: data.get(k) for k, v in expected.items() if data.get(k) != v}
    if wrong:
        raise SystemExit(f"{out} does not match this evaluation ({wrong}); move it aside to rerun it")


def planned_rounds(run: Path, max_rounds: int) -> int:
    """Rounds the runner schedules: min(simulated hours / round length, max_rounds)."""

    config = json.loads((run / "sim" / "simulation_config.json").read_text(encoding="utf-8"))
    time_config = config.get("time_config", {})
    total = (time_config.get("total_simulation_hours", 72) * 60) // time_config.get("minutes_per_round", 30)
    return max(1, min(int(total), max_rounds))


NON_ACTIONS = {"DO_NOTHING"}


def action_counts(run: Path) -> Counter:
    """Agent actions per platform and type. Left out: DO_NOTHING (the System
    One policy logs it, OASIS LLM agents never do, so it would compare
    logging, not behaviour) and round 0 (the scripted initial posts, the same
    in both groups; they dominated the mix of a group whose agents act less)."""

    counts: Counter = Counter()
    for platform in ("twitter", "reddit"):
        path = run / "sim" / platform / "actions.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            action = str(row.get("action_type") or "").upper()
            if "event_type" in row or not action or action in NON_ACTIONS or int(row.get("round", 0)) == 0:
                continue
            counts[f"{platform}:{action}"] += 1
    return counts


def post_rows(run: Path) -> list[dict[str, Any]]:
    """Posts, quotes and comments: round, author name, text."""

    rows: list[dict[str, Any]] = []
    for platform in ("twitter", "reddit"):
        path = run / "sim" / platform / "actions.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "event_type" in row or row.get("action_type") not in ("CREATE_POST", "QUOTE_POST", "CREATE_COMMENT"):
                continue
            args = row.get("action_args") or {}
            text = args.get("quote_content") or args.get("content")
            if text:
                rows.append({
                    "round": int(row.get("round", 0)),
                    # Names match across groups; an id would not, so mark it.
                    "agent": str(row.get("agent_name") or f"id:{row.get('agent_id')}"),
                    "text": str(text),
                })
    return rows


def posts_by_round(run: Path) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    for row in post_rows(run):
        result.setdefault(row["round"], []).append(row["text"])
    return result


def stance_question_for(prepared: dict[str, Any]) -> str:
    """The scenario's question file, or the golden ``STANCE_QUESTION``.

    A prepared dir records the fixture path it was made from; after the
    checkout moves, that path is gone. The fixture is then found by its
    digest in the suite and calibration manifests. It used to fall back to
    the golden question silently, so every non-golden scenario was scored
    on the air-taxi question; an unknown fixture now stops the run.
    """

    if not prepared.get("fixture"):
        return STANCE_QUESTION  # prepared before --fixture: the golden scenario
    fixture = Path(prepared["fixture"])
    if not fixture.is_dir():
        fixture = _fixture_by_digest(prepared.get("fixture_digest"))
        if fixture is None:
            raise SystemExit(
                f"fixture {prepared['fixture']} is gone and digest {prepared.get('fixture_digest')} "
                "is in no manifest; pass --stance-question"
            )
    path = fixture / "stance_question.txt"
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return STANCE_QUESTION


def _fixture_by_digest(digest: str | None) -> Path | None:
    if not digest:
        return None
    fixtures = BACKEND_DIR / "tests" / "fixtures"
    for manifest in (fixtures / "scenarios" / "suite.json", fixtures / "calibration" / "calibration.json"):
        if not manifest.is_file():
            continue
        for row in json.loads(manifest.read_text(encoding="utf-8")).get("scenarios", []):
            if row.get("digest") == digest:
                return BACKEND_DIR / row["path"]
    return None


def scorer_endpoint(name: str) -> tuple[str, str, str]:
    """(base_url, model, prompt_format) of the ``MODEL_POOL`` entry ``name``."""

    import os

    if not os.environ.get("MODEL_POOL", "").strip():
        raise SystemExit(f"--scorer {name}: MODEL_POOL is not set")
    from app.model_pool import load_pool

    for entry in load_pool():
        if entry.name == name:
            return entry.base_url, entry.model, entry.prompt_format
    raise SystemExit(f"--scorer {name}: no MODEL_POOL entry named {name}")


def scorer_client(base_url: str | None, model: str | None, prompt_format: str = "chatml"):
    """A readout client on another model, so the posts are not judged by the
    model that wrote and checked them (#48); None keeps the System One client."""

    if not base_url or not model:
        return None
    from app.system_one.backends import LocalReadoutBackend
    from app.system_one.client import SystemOneClient, load_temperatures

    return SystemOneClient(
        LocalReadoutBackend(
            base_url=base_url, model=model, prompt_format=prompt_format,
            temperatures=load_temperatures(model=model),
        )
    )


def score_posts(
    client,
    texts: list[str],
    cache: dict[str, dict[str, Any]],
    question: str = STANCE_QUESTION,
    workers: int = 8,
) -> None:
    """Fill ``cache[text] = {"unit": 0..1, "levels": [p per stance level]}``."""

    from concurrent.futures import ThreadPoolExecutor

    from app.system_one.models import ScoreQuestion, SystemOneRequest

    todo = sorted({t for t in texts if t not in cache})

    def ask(text: str) -> tuple[str, dict[str, Any]]:
        answer = client.ask(
            SystemOneRequest(
                state=f"貼文：{text[:300]}",
                questions={"s": ScoreQuestion(instructions=question, criteria=STANCE_LEVELS)},
            )
        ).answers["s"]
        levels = [float(answer.probabilities.get(level, 0.0)) for level in STANCE_LEVELS]
        return text, {"unit": answer.score / (len(STANCE_LEVELS) - 1), "levels": levels}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for text, value in pool.map(ask, todo):
            cache[text] = value


WINDOW = 4


def stance_measures(rows: list[dict[str, Any]], cache: dict[str, dict[str, Any]], rounds: int) -> dict[str, Any]:
    """Per-round and windowed curves, per-persona means and the level mix."""

    by_round: dict[int, list[float]] = {}
    by_agent: dict[str, list[float]] = {}
    levels = [0.0] * len(STANCE_LEVELS)
    for row in rows:
        scored = cache[row["text"]]
        by_round.setdefault(row["round"], []).append(scored["unit"])
        by_agent.setdefault(row["agent"], []).append(scored["unit"])
        levels = [a + b for a, b in zip(levels, scored["levels"])]
    curve = [statistics.mean(by_round[r]) if by_round.get(r) else None for r in range(1, rounds + 1)]
    windowed = []
    for start in range(1, rounds + 1, WINDOW):
        values = [v for r in range(start, min(start + WINDOW, rounds + 1)) for v in by_round.get(r, [])]
        windowed.append(statistics.mean(values) if values else None)
    return {
        "stance_curve": curve,
        "stance_curve_windowed": windowed,
        "stance_by_persona": {agent: statistics.mean(v) for agent, v in sorted(by_agent.items())},
        "stance_levels": {level: round(n, 4) for level, n in zip(STANCE_LEVELS, levels)},
    }


BOOTSTRAP_REPS = 1000
PERSONA_DELTA = 0.15
ALPHA = 0.05
MIN_COMMON_PERSONAS = 12  # below this the persona gate is undecidable
MIN_COMMON_SHARE = 0.8


def persona_correlation(x: dict[str, float], y: dict[str, float]) -> float | None:
    common = sorted(set(x) & set(y))
    return pearson([x[k] for k in common], [y[k] for k in common])


def enough_common_personas(a: dict[str, float], b: dict[str, float]) -> tuple[int, bool]:
    """A few shared personas can correlate by chance: require >= 12 and >= 80% of A's."""

    common = len(set(a) & set(b))
    return common, bool(a) and common >= MIN_COMMON_PERSONAS and common >= MIN_COMMON_SHARE * len(a)


def mean_persona(runs: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for run in runs:
        for agent, v in run["stance_by_persona"].items():
            values.setdefault(agent, []).append(v)
    return {agent: statistics.mean(v) for agent, v in values.items()}


def mean_curve(curves: list[list[float | None]]) -> list[float | None]:
    out = []
    for values in zip(*curves):
        present = [v for v in values if v is not None]
        out.append(statistics.mean(present) if present else None)
    return out


def bootstrap_ci(values: list[float], alpha: float = ALPHA) -> tuple[float, float]:
    """Percentile CI: sorted[floor(alpha/2*n)], sorted[ceil((1-alpha/2)*n)-1]."""

    ordered = sorted(values)
    n = len(ordered)
    lo = math.floor(alpha / 2 * n)
    hi = math.ceil((1 - alpha / 2) * n) - 1
    return (float(ordered[lo]), float(ordered[hi]))


def _distinct_pairs(sample: list[dict]) -> list[tuple[dict, dict]]:
    """Pairs whose original run objects differ (same object means same seed)."""

    pairs = []
    for i, j in itertools.combinations(range(len(sample)), 2):
        if sample[i] is not sample[j]:
            pairs.append((sample[i], sample[j]))
    return pairs


def bootstrap_between(
    a_runs: list[dict],
    b_runs: list[dict],
    stat: Callable[[list[dict], list[dict]], float | None],
    reps: int,
    rng: random.Random,
) -> list[float]:
    """Resample A and B with replacement; ``stat(b_sample, a_sample)`` each rep."""

    values: list[float] = []
    if not a_runs or not b_runs:
        return values
    for _ in range(reps):
        a_sample = [rng.choice(a_runs) for _ in range(len(a_runs))]
        b_sample = [rng.choice(b_runs) for _ in range(len(b_runs))]
        value = stat(b_sample, a_sample)
        if value is not None:
            values.append(float(value))
    return values


def bootstrap_within(
    a_runs: list[dict],
    stat: Callable[[list[tuple[dict, dict]]], float | None],
    reps: int,
    rng: random.Random,
) -> list[float]:
    """Resample A with replacement. Pairs only across distinct original seeds.

    A rep with no such pair is skipped.
    """

    values: list[float] = []
    if len(a_runs) < 2:
        return values
    for _ in range(reps):
        sample = [rng.choice(a_runs) for _ in range(len(a_runs))]
        pairs = _distinct_pairs(sample)
        if not pairs:
            continue
        value = stat(pairs)
        if value is not None:
            values.append(float(value))
    return values


def one_sided_p(ba: list[float], aa: list[float]) -> float:
    """Share of reps with ``ba_i <= aa_i``."""

    n = min(len(ba), len(aa))
    if n == 0:
        return 1.0
    return sum(b <= a for b, a in zip(ba, aa)) / n


def _paired_bootstrap(
    a_runs: list[dict],
    b_runs: list[dict],
    between: Callable[[list[dict], list[dict]], float | None],
    within: Callable[[list[tuple[dict, dict]]], float | None],
    reps: int,
    rng: random.Random,
) -> tuple[list[float], list[float]]:
    """One resample of A and B per rep. Skip the rep when A has no cross-seed pair."""

    ba: list[float] = []
    aa: list[float] = []
    if len(a_runs) < 2 or not b_runs:
        return ba, aa
    for _ in range(reps):
        a_sample = [rng.choice(a_runs) for _ in range(len(a_runs))]
        b_sample = [rng.choice(b_runs) for _ in range(len(b_runs))]
        pairs = _distinct_pairs(a_sample)
        if not pairs:
            continue
        b_value = between(b_sample, a_sample)
        a_value = within(pairs)
        if b_value is None or a_value is None:
            continue
        ba.append(float(b_value))
        aa.append(float(a_value))
    return ba, aa


def _mean_js_between(key: str) -> Callable[[list[dict], list[dict]], float | None]:
    def stat(left: list[dict], right: list[dict]) -> float | None:
        vals = [js_divergence(b[key], a[key]) for b in left for a in right]
        return statistics.mean(vals) if vals else None

    return stat


def _mean_js_within(key: str) -> Callable[[list[tuple[dict, dict]]], float | None]:
    def stat(pairs: list[tuple[dict, dict]]) -> float | None:
        vals = [js_divergence(x[key], y[key]) for x, y in pairs]
        return statistics.mean(vals) if vals else None

    return stat


# ------------------------------------------------------------------ gates


def _mean_of(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(values) if values else None


def _content_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    """Mean over runs that produced posts (a run with none has no rate)."""

    values = [r["content"][key] for r in rows if r["content"].get("posts") and r["content"].get(key) is not None]
    return statistics.mean(values) if values else None


def _clean_runs(rows: dict[int, dict[str, Any]]) -> list[dict]:
    return [rows[seed] for seed in sorted(rows) if not rows[seed].get("llm_errors")]


def _js_gate(
    point_b: float | None,
    point_a: float | None,
    ba: list[float],
    aa: list[float],
    *,
    comparable: bool,
) -> dict[str, Any]:
    """One-sided bootstrap gate. Fewer than two clean A runs, or no valid rep, fails."""

    if not comparable or not ba:
        return {
            "b_vs_a": point_b,
            "b_vs_a_ci": None,
            "a_seed_noise": point_a,
            "a_seed_noise_ci": None,
            "p_value": None,
            "alpha": ALPHA,
            "status": "fail",
            "passed": False,
        }
    p_value = one_sided_p(ba, aa)
    status = "fail" if p_value < ALPHA else "pass"
    return {
        "b_vs_a": point_b,
        "b_vs_a_ci": list(bootstrap_ci(ba)),
        "a_seed_noise": point_a,
        "a_seed_noise_ci": list(bootstrap_ci(aa)),
        "p_value": p_value,
        "alpha": ALPHA,
        "status": status,
        "passed": status == "pass",
    }


def _gate_ok(gate: dict[str, Any]) -> bool:
    """``undecidable`` counts as a pass for the recommendation."""

    status = gate.get("status")
    if status == "undecidable":
        return True
    if status in ("pass", "fail"):
        return status == "pass"
    return bool(gate.get("passed"))


def evaluate_groups(
    per_run: dict[str, dict[int, dict[str, Any]]],
    *,
    reps: int = BOOTSTRAP_REPS,
    delta: float = PERSONA_DELTA,
    rng_seed: int = 0,
) -> tuple[dict[str, Any], str, dict[str, Any], list[str]]:
    """G4 v2 gates over the clean runs; runs with model-server errors lost agent
    turns and are excluded (and listed)."""

    excluded = sorted(
        f"{name}_seed{seed}" for name, rows in per_run.items() for seed, row in rows.items() if row.get("llm_errors")
    )
    a_runs = _clean_runs(per_run.get("A", {}))
    b_runs = _clean_runs(per_run.get("B", {}))
    rng = random.Random(rng_seed)

    js_aa = [js_divergence(x["actions"], y["actions"]) for x, y in itertools.combinations(a_runs, 2)]
    js_ba = [js_divergence(b["actions"], a["actions"]) for b in b_runs for a in a_runs]
    def pairwise(fn, key):
        values = [fn(x[key], y[key]) for x, y in itertools.combinations(a_runs, 2)]
        values = [v for v in values if v is not None]
        return statistics.mean(values) if values else None

    # Stance, reported: per-round and windowed curves.
    a_curve = mean_curve([r["stance_curve"] for r in a_runs])
    b_curve = mean_curve([r["stance_curve"] for r in b_runs])
    a_window = mean_curve([r["stance_curve_windowed"] for r in a_runs])
    b_window = mean_curve([r["stance_curve_windowed"] for r in b_runs])
    curve_report = {
        "per_round": {"b_vs_a": pearson(a_curve, b_curve) if a_runs and b_runs else None,
                      "a_seed_pairs_mean": pairwise(pearson, "stance_curve")},
        "windowed": {"b_vs_a": pearson(a_window, b_window) if a_runs and b_runs else None,
                     "a_seed_pairs_mean": pairwise(pearson, "stance_curve_windowed")},
    }
    # Stance, gated: per-persona means and the level mix.
    a_persona, b_persona = mean_persona(a_runs), mean_persona(b_runs)
    persona_ab = persona_correlation(a_persona, b_persona) if a_runs and b_runs else None
    n_common, enough_personas = enough_common_personas(a_persona, b_persona)
    levels_aa = [js_divergence(x["stance_levels"], y["stance_levels"]) for x, y in itertools.combinations(a_runs, 2)]
    levels_ba = [js_divergence(b["stance_levels"], a["stance_levels"]) for b in b_runs for a in a_runs]
    levels_noise = statistics.mean(levels_aa) if levels_aa else None
    levels_b = statistics.mean(levels_ba) if levels_ba else None

    a_decode = _mean_of(a_runs, "decode_per_round")
    b_decode = _mean_of(b_runs, "decode_per_round")
    js_noise = statistics.mean(js_aa) if js_aa else None
    js_b = statistics.mean(js_ba) if js_ba else None
    comparable = len(a_runs) >= 2
    action_ba, action_aa = _paired_bootstrap(
        a_runs, b_runs, _mean_js_between("actions"), _mean_js_within("actions"), reps, rng
    )
    level_ba, level_aa = _paired_bootstrap(
        a_runs, b_runs, _mean_js_between("stance_levels"), _mean_js_within("stance_levels"), reps, rng
    )
    pair_corrs = [
        persona_correlation(x["stance_by_persona"], y["stance_by_persona"])
        for x, y in itertools.combinations(a_runs, 2)
    ]
    pair_corrs = [v for v in pair_corrs if v is not None]
    persona_median = statistics.median(pair_corrs) if pair_corrs else None
    persona_threshold = (persona_median - delta) if persona_median is not None else None
    persona_samples: list[float] = []
    if a_runs and b_runs:
        for _ in range(reps):
            a_sample = [rng.choice(a_runs) for _ in range(len(a_runs))]
            b_sample = [rng.choice(b_runs) for _ in range(len(b_runs))]
            corr = persona_correlation(mean_persona(a_sample), mean_persona(b_sample))
            if corr is not None:
                persona_samples.append(round(float(corr), 12))
    persona_ci = list(bootstrap_ci(persona_samples)) if persona_samples else None
    if n_common < MIN_COMMON_PERSONAS:
        persona_status, persona_passed = "undecidable", None
    elif (
        not enough_personas
        or persona_threshold is None
        or persona_ci is None
        or persona_ci[0] < persona_threshold
    ):
        persona_status, persona_passed = "fail", False
    else:
        persona_status, persona_passed = "pass", True
    # The gate is about the local path fitting a 10 GB card; A may run on a
    # server with larger per-slot contexts (OASIS LLM agents need ~32K+).
    b_vram = [r["vram_peak_mib"] for r in b_runs]
    vram_peak = max(b_vram) if b_vram and None not in b_vram else None  # unmeasured is not a pass
    a_vram = [r["vram_peak_mib"] for r in a_runs if r["vram_peak_mib"] is not None]
    gates = {
        "decode_ratio": {
            "value": (b_decode / a_decode) if a_decode and b_decode is not None else None,
            "threshold": 0.10,
            "passed": bool(a_decode) and b_decode is not None and b_decode <= 0.10 * a_decode,
        },
        "action_js": _js_gate(js_b, js_noise, action_ba, action_aa, comparable=comparable),
        "stance_by_persona": {
            "value": persona_ab,
            "ci": persona_ci,
            "a_seed_pairs_median": persona_median,
            "delta": delta,
            "threshold": persona_threshold,
            "common_personas": n_common,
            "status": persona_status,
            "passed": persona_passed,
        },
        "stance_distribution": _js_gate(levels_b, levels_noise, level_ba, level_aa, comparable=comparable),
        "vram": {
            "peak_mib": vram_peak,
            "a_peak_mib": max(a_vram) if a_vram else None,
            "budget_mib": VRAM_BUDGET_MIB,
            "passed": vram_peak is not None and vram_peak <= VRAM_BUDGET_MIB,
        },
    }
    if all(_gate_ok(g) for g in gates.values()):
        recommendation = "switch"
    elif gates["decode_ratio"]["passed"] and gates["vram"]["passed"] and (
        _gate_ok(gates["action_js"])
        or (_gate_ok(gates["stance_by_persona"]) and _gate_ok(gates["stance_distribution"]))
    ):
        recommendation = "conditional"
    else:
        recommendation = "keep_llm"

    summary = {
        name: {
            "clean_runs": len(rows),
            "actions_per_run": statistics.mean(sum(r["actions"].values()) for r in rows) if rows else None,
            "decode_per_round": _mean_of(rows, "decode_per_round"),
            "round_latency_mean_s": _mean_of(rows, "round_latency_mean_s"),
            "distinct_2": _content_mean(rows, "distinct_2"),
            "entity_mention_rate": _content_mean(rows, "entity_mention_rate"),
        }
        for name, rows in (("A", a_runs), ("B", b_runs))
    }
    summary["stance_curve_A"] = a_curve
    summary["stance_curve_B"] = b_curve
    summary["stance_curve_report"] = curve_report
    summary["stance_by_persona_A"] = a_persona
    summary["stance_by_persona_B"] = b_persona
    return gates, recommendation, summary, excluded


# ------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-a", required=True, help="golden_pipeline prepare --prep-mode llm")
    parser.add_argument("--work-b", required=True, help="golden_pipeline prepare --prep-mode template")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--stance-question", default=None, help="override the scenario stance question")
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--delta", type=float, default=PERSONA_DELTA)
    parser.add_argument("--simulate-only", choices=["A", "B"], default=None,
                        help="only run this group's simulations (e.g. to switch server settings between groups)")
    parser.add_argument("--extraction-recall", type=float, default=None,
                        help="entity recall from #7 (scripts/eval_local_extraction.py)")
    parser.add_argument("--scorer-base-url", default=None,
                        help="score posts with another model's readout (an independent judge, #48)")
    parser.add_argument("--scorer-model", default=None)
    parser.add_argument("--scorer-prompt-format", default="chatml", choices=["chatml", "plain"])
    parser.add_argument("--jobs", type=int, default=1,
                        help="simulations to run at once (independent seeds; the GPU is mostly idle in one run)")
    parser.add_argument("--scorer", default=None,
                        help="MODEL_POOL entry name to score posts with (sets the three --scorer-* options)")
    args = parser.parse_args(argv)
    if args.scorer:
        args.scorer_base_url, args.scorer_model, args.scorer_prompt_format = scorer_endpoint(args.scorer)
    if gp.dotenv_files():
        parser.error("a .env file exists; move it aside (see golden_pipeline.py)")

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    groups = {
        "A": {"work": Path(args.work_a).resolve(), "backend": "llm", "content": None},
        "B": {"work": Path(args.work_b).resolve(), "backend": "system_one", "content": "tiered"},
    }
    # Refuse a mismatched pair before any simulation runs.
    scenario = gp.check_same_scenario(
        *(json.loads((groups[g]["work"] / "prepared.json").read_text(encoding="utf-8")) for g in ("A", "B"))
    )
    runs: dict[str, dict[int, dict[str, Any]]] = {"A": {}, "B": {}}
    chosen = {n: g for n, g in groups.items() if not args.simulate_only or n == args.simulate_only}
    runs.update(run_all(chosen, out, args.seeds, args.rounds, jobs=args.jobs))
    if args.simulate_only:
        return 0

    gp.force_local_config(out)
    from app.system_one.client import get_system_one_client

    client = scorer_client(args.scorer_base_url, args.scorer_model, args.scorer_prompt_format)
    scorer = f"{args.scorer_model} @ {args.scorer_base_url}" if client else "system_one"
    client = client or get_system_one_client()
    cache: dict[str, dict[str, Any]] = {}
    all_rows = {(name, seed): post_rows(info["dir"]) for name in runs for seed, info in runs[name].items()}
    prepared_a = json.loads((groups["A"]["work"] / "prepared.json").read_text(encoding="utf-8"))
    question = args.stance_question or stance_question_for(prepared_a)
    score_posts(client, [row["text"] for rows in all_rows.values() for row in rows], cache, question=question)
    per_run: dict[str, dict[int, dict[str, Any]]] = {"A": {}, "B": {}}
    for name in runs:
        work = groups[name]["work"]
        prepared = json.loads((work / "prepared.json").read_text(encoding="utf-8"))
        names = gp.entity_names(work, prepared)
        for seed, info in runs[name].items():
            summary = info["summary"]
            active = max(summary.get("rounds_logged", {}).values() or [0])
            scheduled = planned_rounds(info["dir"], args.rounds)
            texts = [t for ts in posts_by_round(info["dir"]).values() for t in ts]
            latencies = [
                v.get("mean") for v in summary.get("round_latency_s", {}).values() if v.get("mean") is not None
            ]
            per_run[name][seed] = {
                "decode_tokens": summary["simulation_decode_tokens"],
                "decode_per_round": summary["simulation_decode_tokens"] / scheduled,
                "scheduled_rounds": scheduled,
                "active_rounds": active,
                "round_latency_mean_s": round(statistics.mean(latencies), 2) if latencies else None,
                "elapsed_s": summary["elapsed_s"],
                "llm_errors": llm_errors(info["dir"]),
                "vram_peak_mib": summary["vram_mib"]["peak"],
                "actions": action_counts(info["dir"]),
                "content": gp.content_stats(texts, names),
                **stance_measures(all_rows[(name, seed)], cache, scheduled),
            }

    gates, recommendation, summary, excluded = evaluate_groups(
        per_run, reps=args.bootstrap_reps, delta=args.delta
    )

    def strip(rows):
        return {
            seed: {k: (dict(v) if isinstance(v, Counter) else v) for k, v in row.items()}
            for seed, row in rows.items()
        }

    report = {
        "scenario": scenario,
        "seeds": args.seeds,
        "rounds": args.rounds,
        "groups": {name: strip(rows) for name, rows in per_run.items()},
        "summary": {**summary, "extraction_recall": args.extraction_recall},
        "gate_version": "g4v2",
        "stance_question": question,
        "scorer": scorer,
        "gates": gates,
        "runs_with_llm_errors": excluded,
        "recommendation": recommendation,
    }
    (out / "ab_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "ab_report.md").write_text(render(report), encoding="utf-8")
    print(render(report))
    return 0


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci_text(bounds: Any) -> str:
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return ""
    return f" [{_fmt(bounds[0])}, {_fmt(bounds[1])}]"


def _gate_mark(gate: dict[str, Any]) -> str:
    if gate.get("status") == "undecidable":
        return "無法判讀"
    return "✅" if gate.get("passed") else "❌"


def render(report: dict[str, Any]) -> str:
    s, g = report["summary"], report["gates"]
    labels = {"switch": "切換預設值", "conditional": "有條件切換", "keep_llm": "維持 llm"}
    lines = [
        f"# A/B 忠實度評估（{report.get('scenario', 'golden_scenario')}）",
        "",
        f"- seeds：{', '.join(map(str, report['seeds']))}；每組 {len(report['seeds'])} 次；每次 {report['rounds']} 回合",
        "- A：LLM 決策（LLM 準備）；B：System One 決策＋分層內容（模板／結構化準備）；兩組都用本機圖譜與本機模型",
        "",
        "## 指標",
        "",
        "| 指標 | A | B |",
        "| --- | --- | --- |",
        f"| 閘門採用的 run 數（排除伺服器錯誤） | {s['A'].get('clean_runs', '—')} | {s['B'].get('clean_runs', '—')} |",
        f"| 每 run 的 agent 動作數（不含第 0 回合） | {_fmt(s['A'].get('actions_per_run'), 1)} | {_fmt(s['B'].get('actions_per_run'), 1)} |",
        f"| 每回合 decode tokens | {_fmt(s['A']['decode_per_round'], 1)} | {_fmt(s['B']['decode_per_round'], 1)} |",
        f"| 有動作回合平均延遲 (s) | {_fmt(s['A']['round_latency_mean_s'], 1)} | {_fmt(s['B']['round_latency_mean_s'], 1)} |",
        f"| distinct-2 | {_fmt(s['A']['distinct_2'])} | {_fmt(s['B']['distinct_2'])} |",
        f"| 種子實體出現率 | {_fmt(s['A']['entity_mention_rate'])} | {_fmt(s['B']['entity_mention_rate'])} |",
        f"| 抽取實體召回（#7） | — | {_fmt(s['extraction_recall'])} |",
        "",
        "## 閘門 G4",
        "",
        "| 條件 | 數值 | 門檻 | 結果 |",
        "| --- | --- | --- | --- |",
        f"| B／A 每回合 decode | {_fmt(g['decode_ratio']['value'])} | ≤ 0.10 | {'✅' if g['decode_ratio']['passed'] else '❌'} |",
        f"| 動作分布 JS（B vs A） | {_fmt(g['action_js']['b_vs_a'])}{_ci_text(g['action_js'].get('b_vs_a_ci'))}（A 組間 {_fmt(g['action_js']['a_seed_noise'])}{_ci_text(g['action_js'].get('a_seed_noise_ci'))}，p {_fmt(g['action_js'].get('p_value'))}） | 單尾 p < {g['action_js'].get('alpha', ALPHA)} 判失敗 | {_gate_mark(g['action_js'])} |",
        f"| 各角色平均立場相關（B vs A） | {_fmt(g['stance_by_persona']['value'])}{_ci_text(g['stance_by_persona'].get('ci'))}（中位數 − δ = {_fmt(g['stance_by_persona'].get('threshold'))}；共同角色 {g['stance_by_persona'].get('common_personas', '—')}） | 下界 ≥ 中位數 − δ，共同角色 ≥ {MIN_COMMON_PERSONAS} 且 ≥ 80% | {_gate_mark(g['stance_by_persona'])} |",
        f"| 立場分布 JS（B vs A） | {_fmt(g['stance_distribution']['b_vs_a'])}{_ci_text(g['stance_distribution'].get('b_vs_a_ci'))}（A 組間 {_fmt(g['stance_distribution']['a_seed_noise'])}{_ci_text(g['stance_distribution'].get('a_seed_noise_ci'))}，p {_fmt(g['stance_distribution'].get('p_value'))}） | 單尾 p < {g['stance_distribution'].get('alpha', ALPHA)} 判失敗 | {_gate_mark(g['stance_distribution'])} |",
        f"| VRAM 峰值（B） | {_fmt(g['vram']['peak_mib'])} MiB（A {_fmt(g['vram'].get('a_peak_mib'))} MiB） | ≤ {g['vram']['budget_mib']} MiB | {'✅' if g['vram']['passed'] else '❌'} |",
        "",
        f"## 建議：{labels[report['recommendation']]}",
        "",
    ]
    curves = s.get("stance_curve_report")
    if curves:
        lines += [
            "## 立場曲線（只報告，不作為閘門）",
            "",
            "| 曲線 | B vs A | A 組間平均 |",
            "| --- | --- | --- |",
            f"| 每回合 | {_fmt(curves['per_round']['b_vs_a'])} | {_fmt(curves['per_round']['a_seed_pairs_mean'])} |",
            f"| 每 {WINDOW} 回合 | {_fmt(curves['windowed']['b_vs_a'])} | {_fmt(curves['windowed']['a_seed_pairs_mean'])} |",
            "",
        ]
    if report.get("runs_with_llm_errors"):
        lines += [
            f"⚠️ 下列 run 有模型伺服器錯誤（丟失 agent 回合），已排除在閘門之外：{', '.join(report['runs_with_llm_errors'])}",
            "",
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
