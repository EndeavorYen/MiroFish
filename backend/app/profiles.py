"""Setting profiles (#36): one switch for the local-first stack.

``MIROFISH_PROFILE=local`` runs everything on the local model server with
the zero-decode paths (System One decisions, tiered content, structured
prep, metrics report). ``MIROFISH_PROFILE=local-llm`` keeps everything
local but lets the local LLM decide agent actions (closer to the hosted
behaviour, ~15x the decode of ``local``).

Mode settings (decisions, content, prep, report ...) are only filled when
unset, so ``.env`` can still change them. Service endpoints are different:
a local profile must not call external APIs, so a hosted ``LLM_BASE_URL`` or
``EMBED_BASE_URL`` (e.g. the cloud values ``.env.example`` ships with) is
replaced by the local service, with a warning, and ``LLM_BOOST_*`` (a second
hosted endpoint for Reddit agents) is removed. A local URL on another port
or model name is kept, and System One follows the resolved LLM settings
unless set on its own.

Unset means no profile and no change; an unknown name is an error.
"""

from __future__ import annotations

import logging
import os
from collections.abc import MutableMapping
from urllib.parse import urlparse

logger = logging.getLogger("mirofish.profiles")

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}
LOCAL_LLM = {
    "LLM_API_KEY": "local",
    "LLM_BASE_URL": "http://127.0.0.1:8000/v1",
    "LLM_MODEL_NAME": "qwen3.5-4b",
}
LOCAL_EMBED = {
    "EMBED_BASE_URL": "http://127.0.0.1:8001/v1",
    "EMBED_MODEL_NAME": "intfloat/multilingual-e5-small",
}
DROPPED = ("LLM_BOOST_API_KEY", "LLM_BOOST_BASE_URL", "LLM_BOOST_MODEL_NAME")
_LOCAL_MODES = {
    "SYSTEM_ONE_BACKEND": "local",
    "GRAPH_BACKEND": "local",
    "GRAPH_EXTRACTOR": "local",
    "GRAPH_EMBEDDER": "http",
    "LOCAL_NER": "candidates",
}

PROFILES: dict[str, dict[str, str]] = {
    "local": {
        **_LOCAL_MODES,
        "SIM_DECISION_BACKEND": "system_one",
        "CONTENT_MODE": "tiered",
        "ONTOLOGY_MODE": "template",
        "PROFILE_MODE": "structured",
        "SIM_CONFIG_MODE": "structured",
        "REPORT_MODE": "metrics",
    },
    "local-llm": {
        **_LOCAL_MODES,
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


def is_local_url(url: str | None) -> bool:
    return (urlparse(url or "").hostname or "") in LOCAL_HOSTS


def _blank(env: MutableMapping[str, str], key: str) -> bool:
    return not (env.get(key) or "").strip()


def _local_service(env: MutableMapping[str, str], defaults: dict[str, str], url_key: str, applied: dict) -> None:
    """Keep a local endpoint the user set; replace a hosted one; fill blanks."""

    if not _blank(env, url_key) and not is_local_url(env[url_key]):
        logger.warning(
            "%s=%s is not local; MIROFISH_PROFILE uses %s instead", url_key, env[url_key], defaults[url_key]
        )
        for key, value in defaults.items():
            env[key] = value
            applied[key] = value
        return
    for key, value in defaults.items():
        if _blank(env, key):
            env[key] = value
            applied[key] = value


def apply_profile(environ: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """Apply ``MIROFISH_PROFILE``; returns what was set (removed keys map to "")."""

    env = os.environ if environ is None else environ
    name = (env.get(ENV_VAR) or "").strip().lower()
    if not name:
        return {}
    if name not in PROFILES:
        raise ValueError(f"{ENV_VAR} must be one of {sorted(PROFILES)}, got {name!r}")
    applied: dict[str, str] = {}
    _local_service(env, LOCAL_LLM, "LLM_BASE_URL", applied)
    _local_service(env, LOCAL_EMBED, "EMBED_BASE_URL", applied)
    for key in DROPPED:
        if key in env:
            env.pop(key)
            applied[key] = ""
    # System One reads the same local server unless configured on its own.
    for key, source in (("SYSTEM_ONE_BASE_URL", "LLM_BASE_URL"), ("SYSTEM_ONE_MODEL", "LLM_MODEL_NAME")):
        if _blank(env, key) or (key == "SYSTEM_ONE_BASE_URL" and not is_local_url(env[key])):
            env[key] = env[source]
            applied[key] = env[source]
    for key, value in PROFILES[name].items():
        if _blank(env, key):
            env[key] = value
            applied[key] = value
    return applied
