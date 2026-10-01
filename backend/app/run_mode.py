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


def process_profile() -> str:
    """The backend's own ``MIROFISH_PROFILE``, normalised ('' when unset)."""

    import os

    return (os.environ.get("MIROFISH_PROFILE") or "").strip().lower()


def local_backend() -> bool:
    """A run may choose its mode only on a backend running a local profile:
    the mode settings change per run, the service endpoints do not."""

    return process_profile() in profiles()


@contextmanager
def use(profile: str | None) -> Iterator[None]:
    """Within the block, the mode settings are ``profile``'s. ``None``, an
    unknown profile, or the backend's own profile: the process's settings
    (which keep the .env overrides of that profile)."""

    from .profiles import PROFILES

    profile = (profile or "").strip().lower()
    settings = None
    if profile in PROFILES and profile != process_profile():
        # MIROFISH_PROFILE too: the simulation process applies it again.
        settings = {**profile_settings(profile), "MIROFISH_PROFILE": profile}
    token = _current.set(settings)
    try:
        yield
    finally:
        _current.reset(token)


def get(key: str) -> Any:
    """The run's value: a string, ``None`` (the profile leaves it unset), or
    ``_UNSET`` outside a run (use the process's value)."""

    settings = _current.get()
    if settings is None or key not in MODE_KEYS:
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
    """``target`` running with the current run's mode: give it to a new
    thread, which would otherwise start without it. Only the mode is carried;
    a copy of the whole context would also carry the finished request's."""

    settings = _current.get()

    def run(*args: Any, **kwargs: Any) -> Any:
        context = contextvars.Context()
        context.run(_current.set, settings)
        return context.run(target, *args, **kwargs)

    return run


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
