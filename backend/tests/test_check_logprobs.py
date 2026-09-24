"""Tests for scripts/check_logprobs.py."""

from __future__ import annotations

import pytest

from scripts.check_logprobs import build_logprob_request, extract_candidate_logprobs


def _response(top_logprobs):
    return {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {"token": "x", "top_logprobs": top_logprobs},
                    ]
                }
            }
        ]
    }


def test_extract_candidates_found():
    response = _response(
        [
            {"token": "A", "logprob": -0.1},
            {"token": " B", "logprob": -2.3},
            {"token": "C", "logprob": -4.0},
        ]
    )
    assert extract_candidate_logprobs(response, ["A", "B"]) == {"A": -0.1, "B": -2.3}


def test_extract_candidate_missing_is_none():
    response = _response(
        [
            {"token": "A", "logprob": -0.1},
            {"token": " B", "logprob": -2.3},
            {"token": "C", "logprob": -4.0},
        ]
    )
    assert extract_candidate_logprobs(response, ["D"]) == {"D": None}


def test_build_request_caps_top_k():
    with pytest.raises(ValueError):
        build_logprob_request("m", "p", top_k=65)
    request = build_logprob_request("m", "p", top_k=64)
    assert request["max_tokens"] == 1
    assert request["logprobs"] is True
    assert request["top_logprobs"] == 64
