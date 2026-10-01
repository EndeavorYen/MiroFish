"""A run's own mode (#66): local, local-hybrid or local-llm per run.

``MIROFISH_PROFILE`` fills the environment once at start-up. The settings
that differ between the local profiles (how the preparation, the decisions,
the content and the report are made) can instead come from the run that is
being executed, so one backend can confirm a local run with local-llm.

``use(profile)`` sets them for the current thread's context: the runs
orchestrator wraps a run in it. ``Config`` reads its mode attributes through
``get``; ``bind`` keeps it for a new thread (the prepare and report
endpoints work in threads); ``subprocess_env`` puts them into the
simulation process's environment. Outside a run nothing changes.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Callable, Iterator, MutableMapping

# The settings the local profiles set differently (app/profiles.py).
MODE_KEYS = (
    "SIM_DECISION_BACKEND",
    "CONTENT_MODE",
    "ONTOLOGY_MODE",
    "PROFILE_MODE",
    "SIM_CONFIG_MODE",
    "REPORT_MODE",
    "SIM_AGENT_CONTEXT_TOKENS",
)

_UNSET = object()
_current: contextvars.ContextVar[dict[str, str | None] | None] = contextvars.ContextVar("run_mode", default=None)


def profile_settings(profile: str) -> dict[str, str | None]:
    """The mode settings of a profile; ``None``: the profile leaves it unset."""

    from .profiles import PROFILES

    values = PROFILES[profile]
    return {key: values.get(key) for key in MODE_KEYS}


def profiles() -> tuple[str, ...]:
    from .profiles import PROFILES

    return tuple(PROFILES)


@contextmanager
def use(profile: str | None) -> Iterator[None]:
    """Within the block, the mode settings are ``profile``'s. ``None`` or an
    unknown profile: the process's own settings."""

    from .profiles import PROFILES

    token = _current.set(profile_settings(profile) if profile in PROFILES else None)
    try:
        yield
    finally:
        _current.reset(token)


def get(key: str) -> Any:
    """The run's value: a string, ``None`` (the profile leaves it unset), or
    ``_UNSET`` outside a run (use the process's value)."""

    settings = _current.get()
    if settings is None or key not in settings:
        return _UNSET
    return settings[key]


def setting(key: str, default: str) -> str:
    """The run's value of ``key``, else the environment's, else ``default``."""

    import os

    value = get(key)
    if value is _UNSET:
        value = os.environ.get(key)
    return (value or default).strip().lower()


def active() -> bool:
    return _current.get() is not None


def bind(target: Callable[..., Any]) -> Callable[..., Any]:
    """``target`` running in a copy of the current context: give it to a new
    thread, which would otherwise start without the run's mode."""

    context = contextvars.copy_context()
    return lambda *args, **kwargs: context.run(target, *args, **kwargs)


def subprocess_env(env: MutableMapping[str, str]) -> MutableMapping[str, str]:
    """``env`` with the run's mode settings (unset ones removed)."""

    settings = _current.get()
    if settings is not None:
        for key, value in settings.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
    return env
