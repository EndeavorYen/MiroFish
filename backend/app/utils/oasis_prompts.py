"""Readable (non-escaped) JSON in OASIS agent observations (#28).

``oasis.social_agent.agent_environment`` renders the feed, groups and
messages with ``json.dumps(...)`` and the default ``ensure_ascii=True``, so
every Chinese character reaches the model as a ``\\uXXXX`` escape: six
characters and several tokens instead of about one. A 24-post Chinese feed
alone came to ~10K characters, and a single observation overflowed an 8K
context. ``install_unicode_observations`` makes that module's ``json.dumps``
default to ``ensure_ascii=False``; the content is unchanged.
"""

from __future__ import annotations

import json
from typing import Any


class _UnicodeJson:
    """The json module, with dumps defaulting to ensure_ascii=False."""

    @staticmethod
    def dumps(obj: Any, **kwargs: Any) -> str:
        kwargs.setdefault("ensure_ascii", False)
        return json.dumps(obj, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(json, name)


def install_unicode_observations() -> None:
    """Idempotent; affects only OASIS's agent_environment module."""

    from oasis.social_agent import agent_environment

    if not isinstance(agent_environment.json, _UnicodeJson):
        agent_environment.json = _UnicodeJson()
