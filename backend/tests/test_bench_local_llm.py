"""Tests for scripts/bench_local_llm.py."""

from __future__ import annotations

import pytest

from scripts.bench_local_llm import (
    compute_throughput,
    parse_nvidia_smi_mib,
    prefix_cache_reduction,
)


def test_compute_throughput():
    assert compute_throughput(
        prompt_tokens=2000,
        completion_tokens=101,
        ttft_s=0.5,
        total_s=2.5,
    ) == {"prefill_tps": 4000.0, "decode_tps": 50.0}


def test_prefix_cache_reduction():
    assert prefix_cache_reduction(0.8, 0.2) == pytest.approx(0.75)


def test_parse_nvidia_smi():
    assert parse_nvidia_smi_mib("9123\n") == 9123
    assert parse_nvidia_smi_mib("") is None
