"""Read candidate-token logprobs from an OpenAI-compatible chat endpoint."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from typing import Any


def _get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def extract_candidate_logprobs(response: Any, candidates: list[str]) -> dict[str, float | None]:
    """Return each candidate's best logprob from the first generated token.

    Tokens are compared after ``strip()``. When one candidate matches more
    than one top-k token, the larger logprob is kept. Candidates absent from
    the top-k list map to ``None``.
    """
    choices = _get(response, "choices") or []
    logprobs = _get(choices[0], "logprobs") if choices else None
    content = _get(logprobs, "content") or []
    top_logprobs = _get(content[0], "top_logprobs") if content else None

    best: dict[str, float] = {}
    for item in top_logprobs or []:
        token = str(_get(item, "token") or "").strip()
        logprob = _get(item, "logprob")
        if logprob is None or token == "":
            continue
        previous = best.get(token)
        if previous is None or logprob > previous:
            best[token] = float(logprob)
    return {candidate: best.get(candidate.strip()) for candidate in candidates}


def build_logprob_request(model: str, prompt: str, top_k: int = 20) -> dict[str, Any]:
    """Build a one-token chat request that asks for ``top_k`` logprobs."""
    if top_k > 64:
        raise ValueError("top_k must be <= 64")
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": top_k,
    }


def _post_json(url: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read logprobs for candidate tokens from a local chat model"
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--candidates", required=True, help="Comma-separated candidate tokens")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--api-key", default="local")
    args = parser.parse_args(argv)

    candidates = [part.strip() for part in args.candidates.split(",")]
    payload = build_logprob_request(args.model, args.prompt, top_k=args.top_k)
    url = args.base_url.rstrip("/") + "/chat/completions"
    try:
        response = _post_json(url, payload, args.api_key)
    except urllib.error.URLError as exc:
        print(str(exc), flush=True)
        return 1

    found = extract_candidate_logprobs(response, candidates)
    print(json.dumps(found, sort_keys=True, ensure_ascii=False, indent=2))
    if any(value is None for value in found.values()):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
