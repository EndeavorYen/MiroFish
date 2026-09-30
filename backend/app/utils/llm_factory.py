"""One place that builds OpenAI-compatible clients for the LLM (#68).

Every service used to call ``OpenAI(...)`` itself, each resolving
``Config.LLM_*`` and the missing-key check on its own. The per-use timeouts
stay explicit because they differ on purpose:

- ``PREP``: preparation (profiles, simulation config, ontology) and the
  ReportAgent use the SDK defaults (long requests, SDK retries);
- ``SUMMARY``: the metrics report's one short summary, 120 s;
- ``CONTENT``: tiered post content inside a round, 30 s and no retries, so a
  stalled server falls back to a template instead of stalling the round.
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from ..config import Config


@dataclass(frozen=True)
class ClientPolicy:
    timeout: float | None = None  # None: the SDK default
    max_retries: int | None = None  # None: the SDK default


PREP = ClientPolicy()
SUMMARY = ClientPolicy(timeout=120)
CONTENT = ClientPolicy(timeout=30, max_retries=0)


def make_llm_client(
    policy: ClientPolicy = PREP,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    require_key: bool = True,
) -> OpenAI:
    """An ``OpenAI`` client for ``Config.LLM_*`` unless overridden.

    ``require_key=False`` sends ``"local"`` when no key is set (a local
    server ignores it); otherwise a missing key raises ``ValueError``.
    """

    key = api_key or Config.LLM_API_KEY
    if not key:
        if require_key:
            raise ValueError("LLM_API_KEY 未配置")
        key = "local"
    kwargs = {"api_key": key, "base_url": base_url or Config.LLM_BASE_URL}
    if policy.timeout is not None:
        kwargs["timeout"] = policy.timeout
    if policy.max_retries is not None:
        kwargs["max_retries"] = policy.max_retries
    return OpenAI(**kwargs)
