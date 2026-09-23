"""
LLM Usage metrics tracking module.

Tracks prompt tokens, completion tokens, latency, and stage per LLM call,
appending structured records to `metrics/llm_usage.jsonl`.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional, Set

logger = logging.getLogger("mirofish.llm_usage")

VALID_STAGES: Set[str] = {
    "ontology",
    "graph_build",
    "profile",
    "sim_config",
    "simulation",
    "report",
    "interview",
}

_current_stage: ContextVar[Optional[str]] = ContextVar("_current_stage", default=None)
_current_metrics_dir: ContextVar[Optional[str]] = ContextVar("_current_metrics_dir", default=None)
_file_lock = threading.Lock()


def get_current_stage() -> Optional[str]:
    """Get the currently active stage from context."""
    return _current_stage.get()


def get_effective_metrics_dir() -> Optional[str]:
    """
    Get the effective metrics directory.
    Environment variable MIROFISH_METRICS_DIR has top priority.
    """
    env_dir = os.environ.get("MIROFISH_METRICS_DIR")
    if env_dir:
        return env_dir
    return _current_metrics_dir.get()


@contextmanager
def usage_stage(stage: str, metrics_dir: Optional[str] = None):
    """
    Context manager to set current stage and optional metrics directory.

    Args:
        stage: One of the 7 valid stages.
        metrics_dir: Target directory or path for metrics output.
    """
    if stage not in VALID_STAGES:
        raise ValueError(
            f"Invalid stage '{stage}'. Must be one of: {sorted(VALID_STAGES)}"
        )

    token_stage = _current_stage.set(stage)
    token_dir = None
    if metrics_dir is not None:
        token_dir = _current_metrics_dir.set(metrics_dir)

    try:
        yield
    finally:
        _current_stage.reset(token_stage)
        if token_dir is not None:
            _current_metrics_dir.reset(token_dir)


def extract_usage_tokens(response: Any) -> tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) from various response shapes."""
    if response is None:
        return 0, 0

    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    # If response has a raw attribute with usage (e.g., camel wrappers)
    if usage is None and hasattr(response, "raw"):
        raw = getattr(response, "raw", None)
        usage = getattr(raw, "usage", None)
        if usage is None and isinstance(raw, dict):
            usage = raw.get("usage")

    if usage is None:
        return 0, 0

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    if prompt_tokens is None and isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
    prompt_tokens = int(prompt_tokens or 0)

    completion_tokens = getattr(usage, "completion_tokens", None)
    if completion_tokens is None and isinstance(usage, dict):
        completion_tokens = usage.get("completion_tokens")
    completion_tokens = int(completion_tokens or 0)

    return prompt_tokens, completion_tokens


def record_usage(
    response: Any,
    latency_ms: float,
    *,
    stage: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Record an LLM call usage entry into `llm_usage.jsonl`.

    Args:
        response: OpenAI or camel response object or dict.
        latency_ms: Execution duration in milliseconds.
        stage: Explicit stage override or None to read from context.
        model: Optional model name.

    Returns:
        The recorded entry dict, or None if no metrics directory was configured.
    """
    metrics_target = get_effective_metrics_dir()
    if not metrics_target:
        # No metrics directory configured: no-op
        return None

    active_stage = stage or get_current_stage() or "simulation"
    if active_stage not in VALID_STAGES:
        active_stage = "simulation"

    prompt_tokens, completion_tokens = extract_usage_tokens(response)

    resolved_model = model
    if not resolved_model:
        resolved_model = getattr(response, "model", None)
        if not resolved_model and isinstance(response, dict):
            resolved_model = response.get("model")
    if not resolved_model:
        resolved_model = "unknown"

    entry: Dict[str, Any] = {
        "stage": active_stage,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": round(float(latency_ms), 2),
        "model": str(resolved_model),
        "ts": datetime.now().isoformat(),
    }

    # Determine file path
    if metrics_target.endswith(".jsonl"):
        file_path = metrics_target
        out_dir = os.path.dirname(file_path)
    else:
        out_dir = metrics_target
        file_path = os.path.join(out_dir, "llm_usage.jsonl")

    try:
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with _file_lock:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to record LLM usage metrics to {file_path}: {e}")

    return entry


class WrappedCamelModel:
    """Wrapper around a camel-ai ModelBackend to capture token usage and latency."""

    def __init__(self, inner_model: Any, default_stage: str = "simulation"):
        self._inner_model = inner_model
        self._default_stage = default_stage

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner_model, name)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        stage = get_current_stage() or self._default_stage
        start_time = time.perf_counter()
        response = self._inner_model.run(*args, **kwargs)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        model_name = getattr(self._inner_model, "model_type", None) or getattr(response, "model", None)
        record_usage(response, latency_ms=latency_ms, stage=stage, model=model_name)
        return response

    async def arun(self, *args: Any, **kwargs: Any) -> Any:
        stage = get_current_stage() or self._default_stage
        start_time = time.perf_counter()
        response = await self._inner_model.arun(*args, **kwargs)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        model_name = getattr(self._inner_model, "model_type", None) or getattr(response, "model", None)
        record_usage(response, latency_ms=latency_ms, stage=stage, model=model_name)
        return response

    def step(self, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self._inner_model, "step"):
            stage = get_current_stage() or self._default_stage
            start_time = time.perf_counter()
            response = self._inner_model.step(*args, **kwargs)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            model_name = getattr(self._inner_model, "model_type", None) or getattr(response, "model", None)
            record_usage(response, latency_ms=latency_ms, stage=stage, model=model_name)
            return response
        raise AttributeError(f"'{type(self._inner_model).__name__}' object has no attribute 'step'")


def wrap_camel_model(model_backend: Any, default_stage: str = "simulation") -> Any:
    """Wrap a camel-ai model backend if not already wrapped."""
    if isinstance(model_backend, WrappedCamelModel):
        return model_backend
    return WrappedCamelModel(model_backend, default_stage=default_stage)
