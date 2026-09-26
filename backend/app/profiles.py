"""Setting profiles (#36): one switch for the local-first stack.

``MIROFISH_PROFILE=local`` runs everything on the local model server with
the zero-decode paths (System One decisions, tiered content, structured
prep, metrics report). ``MIROFISH_PROFILE=local-llm`` keeps everything
local but lets the local LLM decide agent actions (closer to the hosted
behaviour, ~20x the decode of ``local``).

A profile only fills in variables that are not set, so anything in the
environment or ``.env`` still wins. Unset means no profile and no change;
an unknown name is an error.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping

_LOCAL_SERVICES = {
    "LLM_API_KEY": "local",
    "LLM_BASE_URL": "http://127.0.0.1:8000/v1",
    "LLM_MODEL_NAME": "qwen3.5-4b",
    "SYSTEM_ONE_BACKEND": "local",
    "SYSTEM_ONE_BASE_URL": "http://127.0.0.1:8000/v1",
    "SYSTEM_ONE_MODEL": "qwen3.5-4b",
    "EMBED_BASE_URL": "http://127.0.0.1:8001/v1",
    "EMBED_MODEL_NAME": "intfloat/multilingual-e5-small",
    "GRAPH_BACKEND": "local",
    "GRAPH_EXTRACTOR": "local",
    "GRAPH_EMBEDDER": "http",
    "LOCAL_NER": "candidates",
}

PROFILES: dict[str, dict[str, str]] = {
    "local": {
        **_LOCAL_SERVICES,
        "SIM_DECISION_BACKEND": "system_one",
        "CONTENT_MODE": "tiered",
        "ONTOLOGY_MODE": "template",
        "PROFILE_MODE": "structured",
        "SIM_CONFIG_MODE": "structured",
        "REPORT_MODE": "metrics",
    },
    "local-llm": {
        **_LOCAL_SERVICES,
        "SIM_DECISION_BACKEND": "llm",
        # Agent memory within an 8K-per-slot llama-server (#28).
        "SIM_AGENT_CONTEXT_TOKENS": "3072",
        "ONTOLOGY_MODE": "llm",
        "PROFILE_MODE": "llm",
        "SIM_CONFIG_MODE": "llm",
        "REPORT_MODE": "metrics",
    },
}
ENV_VAR = "MIROFISH_PROFILE"


def apply_profile(environ: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """Fill unset variables from ``MIROFISH_PROFILE``; returns what was set."""

    env = os.environ if environ is None else environ
    name = (env.get(ENV_VAR) or "").strip().lower()
    if not name:
        return {}
    if name not in PROFILES:
        raise ValueError(f"{ENV_VAR} must be one of {sorted(PROFILES)}, got {name!r}")
    applied = {}
    for key, value in PROFILES[name].items():
        if not (env.get(key) or "").strip():
            env[key] = value
            applied[key] = value
    return applied
