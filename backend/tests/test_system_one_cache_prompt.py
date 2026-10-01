"""SYSTEM_ONE_CACHE_PROMPT=0 for bit-exact replays (#82)."""

import pytest

from app.system_one.backends import LocalReadoutBackend


@pytest.mark.parametrize("value, expected", [(None, True), ("1", True), ("0", False), ("false", False), ("off", False)])
def test_the_readout_reuses_the_prefix_cache_unless_told_not_to(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("SYSTEM_ONE_CACHE_PROMPT", raising=False)
    else:
        monkeypatch.setenv("SYSTEM_ONE_CACHE_PROMPT", value)
    backend = LocalReadoutBackend(base_url="http://127.0.0.1:8000/v1", model="m", post=lambda payload: {})
    assert backend._payload("prompt")["cache_prompt"] is expected
