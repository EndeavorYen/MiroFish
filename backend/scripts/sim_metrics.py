"""
Simulation Metrics Calculation Script.

Reads actions.jsonl (from PlatformActionLogger) and computes:
- Per-round action_type distribution
- Active agents count per round
- Post count per round
- Distinct-2 bigram ratio for post contents per round
- Overall summary statistics

Output format is deterministic JSON:
json.dumps(..., sort_keys=True, ensure_ascii=False, indent=2)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words/characters.
    Treats Chinese characters individually and Latin/digit words as chunks.
    """
    if not text:
        return []
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", text.lower())


def compute_distinct_2(texts: List[str]) -> float:
    """
    Calculate distinct-2 ratio: unique bigrams / total bigrams.
    Returns 0.0 if total bigrams == 0.
    """
    bigrams: List[Tuple[str, str]] = []
    for text in texts:
        tokens = tokenize_text(text)
        for i in range(len(tokens) - 1):
            bigrams.append((tokens[i], tokens[i + 1]))

    if not bigrams:
        return 0.0

    return round(len(set(bigrams)) / len(bigrams), 4)


def extract_post_text(action_args: Dict[str, Any]) -> Optional[str]:
    """Extract post text content from action arguments."""
    if not isinstance(action_args, dict):
        return None
    for key in ("content", "text", "body", "message"):
        val = action_args.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def is_post_action(action_type: str, action_args: Dict[str, Any]) -> bool:
    """Return True if the action represents a new post or tweet."""
    norm_type = str(action_type).upper()
    if norm_type in {"CREATE_POST", "POST", "CREATE_TWEET", "TWEET"}:
        return True
    return False


def load_action_entries(source: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Load action records from a jsonl file path or list of dicts.
    Skips rows that contain an 'event_type' field (round_start, round_end, etc.).
    """
    entries: List[Dict[str, Any]] = []

    if isinstance(source, list):
        for item in source:
            if isinstance(item, dict) and "event_type" not in item:
                entries.append(item)
        return entries

    path = str(source)
    # If path is a directory, search for actions.jsonl files
    files_to_read: List[str] = []
    if os.path.isdir(path):
        root_actions = os.path.join(path, "actions.jsonl")
        if os.path.isfile(root_actions):
            files_to_read.append(root_actions)
        else:
            for sub in ("twitter", "reddit"):
                candidate = os.path.join(path, sub, "actions.jsonl")
                if os.path.isfile(candidate):
                    files_to_read.append(candidate)
    elif os.path.isfile(path):
        files_to_read.append(path)
    else:
        raise FileNotFoundError(f"Actions file or directory not found: {path}")

    for file_path in files_to_read:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if isinstance(record, dict) and "event_type" not in record:
                        entries.append(record)
                except json.JSONDecodeError:
                    continue

    return entries


def compute_sim_metrics(source: Union[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Compute simulation metrics from action entries.

    Returns deterministic dict containing:
    - rounds: list of round dicts sorted by round number
    - total_rounds: count of unique rounds
    - summary: overall statistics (total_actions, total_posts, overall_distinct_2, etc.)
    """
    entries = load_action_entries(source)

    # Group actions by round
    rounds_data: Dict[int, Dict[str, Any]] = {}
    all_post_texts: List[str] = []
    all_agent_ids: set[Any] = set()

    for entry in entries:
        r = entry.get("round")
        if r is None:
            continue
        try:
            round_num = int(r)
        except (ValueError, TypeError):
            continue

        if round_num not in rounds_data:
            rounds_data[round_num] = {
                "active_agents": set(),
                "action_type_distribution": {},
                "post_texts": [],
            }

        agent_id = entry.get("agent_id")
        if agent_id is not None:
            rounds_data[round_num]["active_agents"].add(agent_id)
            all_agent_ids.add(agent_id)

        action_type = entry.get("action_type") or "UNKNOWN"
        dist = rounds_data[round_num]["action_type_distribution"]
        dist[action_type] = dist.get(action_type, 0) + 1

        action_args = entry.get("action_args") or {}
        if is_post_action(action_type, action_args):
            text = extract_post_text(action_args)
            if text:
                rounds_data[round_num]["post_texts"].append(text)
                all_post_texts.append(text)

    sorted_rounds = sorted(rounds_data.keys())
    round_summaries: List[Dict[str, Any]] = []

    for round_num in sorted_rounds:
        r_info = rounds_data[round_num]
        post_texts = r_info["post_texts"]
        # Ensure action distribution keys are sorted
        sorted_dist = {
            k: r_info["action_type_distribution"][k]
            for k in sorted(r_info["action_type_distribution"].keys())
        }

        round_summaries.append({
            "action_type_distribution": sorted_dist,
            "active_agents_count": len(r_info["active_agents"]),
            "distinct_2": compute_distinct_2(post_texts),
            "post_count": len(post_texts),
            "round": round_num,
        })

    summary = {
        "overall_active_agents_count": len(all_agent_ids),
        "overall_distinct_2": compute_distinct_2(all_post_texts),
        "total_actions": len(entries),
        "total_posts": len(all_post_texts),
    }

    return {
        "rounds": round_summaries,
        "summary": summary,
        "total_rounds": len(sorted_rounds),
    }


def format_metrics_json(metrics: Dict[str, Any]) -> str:
    """Format metrics dictionary into deterministic, sorted JSON."""
    return json.dumps(metrics, sort_keys=True, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute simulation metrics from actions.jsonl")
    parser.add_argument("actions_path", help="Path to actions.jsonl file or simulation directory")
    parser.add_argument("-o", "--output", help="Optional output JSON path (default: stdout)")
    args = parser.parse_args()

    metrics = compute_sim_metrics(args.actions_path)
    output_json = format_metrics_json(metrics)

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json + "\n")
        print(f"Metrics saved to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
