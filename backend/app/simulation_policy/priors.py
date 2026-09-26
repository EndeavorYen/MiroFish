"""Action priors for the System One decision tree (#26).

``action_priors.json`` holds, per platform, a weight per option at each tree
node (node key: the option path joined by "/", "" for the root). ``ask_tree``
samples from readout x weight, renormalised. The weights are fitted offline
by ``scripts/fit_action_priors.py`` from LLM-decision runs and unweighted
System One readouts on calibration seeds that the A/B evaluation does not
use.

``SIM_ACTION_PRIORS``: unset or ``default`` = the packaged file (if present),
``off`` = no priors, anything else = a path to another priors file.
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from .taxonomy import Taxonomy

PRIORS_PATH = Path(__file__).with_name("action_priors.json")
ENV_VAR = "SIM_ACTION_PRIORS"


def load_action_priors(setting: str | None = None) -> dict[str, dict[str, dict[str, float]]]:
    """``{platform: {node_key: {option: weight}}}``; empty when off or missing."""

    value = (os.environ.get(ENV_VAR, "") if setting is None else setting).strip()
    if value.lower() == "off":
        return {}
    path = PRIORS_PATH if value in ("", "default") else Path(value)
    if not path.exists():
        if value in ("", "default"):
            return {}
        raise FileNotFoundError(f"{ENV_VAR} points to a missing file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    priors = data.get("priors") or {}
    for platform, nodes in priors.items():
        for node, weights in nodes.items():
            for option, weight in weights.items():
                if isinstance(weight, bool) or not isinstance(weight, (int, float)) \
                        or not math.isfinite(weight) or weight < 0:
                    raise ValueError(f"bad prior {platform}/{node}/{option}: {weight!r}")
    return priors


def load_extra_action_rates(setting: str | None = None) -> dict[str, float]:
    """``{platform: rate}`` from the same file (``extra_action_rate``); empty when off."""

    value = (os.environ.get(ENV_VAR, "") if setting is None else setting).strip()
    if value.lower() == "off":
        return {}
    path = PRIORS_PATH if value in ("", "default") else Path(value)
    if not path.exists():
        return {}
    rates = json.loads(path.read_text(encoding="utf-8")).get("extra_action_rate") or {}
    for platform, rate in rates.items():
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 <= rate < 1:
            raise ValueError(f"bad extra_action_rate for {platform}: {rate!r}")
    return {platform: float(rate) for platform, rate in rates.items()}


def _attach(node: dict[str, Any], path: tuple[str, ...], priors: dict[str, dict[str, float]]) -> None:
    weights = priors.get("/".join(path))
    if weights:
        options = set(node["question"]["criteria"])
        unknown = set(weights) - options
        if unknown:
            raise ValueError(f"priors for {'/'.join(path) or 'root'} name unknown options {sorted(unknown)}")
        node["priors"] = dict(weights)
    for key, child in (node.get("children") or {}).items():
        _attach(child, path + (key,), priors)


def with_priors(taxonomy: Taxonomy, priors: dict[str, dict[str, float]] | None) -> Taxonomy:
    """A copy of ``taxonomy`` whose tree nodes carry the priors."""

    if not priors:
        return taxonomy
    tree = copy.deepcopy(taxonomy.tree)
    known: set[str] = set()

    def keys(node: dict[str, Any], path: tuple[str, ...]) -> None:
        known.add("/".join(path))
        for key, child in (node.get("children") or {}).items():
            keys(child, path + (key,))

    keys(tree, ())
    unknown = set(priors) - known
    if unknown:
        raise ValueError(f"priors name unknown tree nodes {sorted(unknown)} for {taxonomy.platform}")
    _attach(tree, (), priors)
    return replace(taxonomy, tree=tree)
