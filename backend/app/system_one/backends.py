"""System One backends.

``LocalReadoutBackend`` calls an OpenAI-compatible ``/v1/completions`` on the
already-running local model server (vLLM or llama.cpp ``llama-server``) with
``max_tokens=1`` and ``logprobs=top_k``. It never starts a server process.

Which logprobs come back:

* vLLM V1 defaults to ``logprobs_mode=raw_logprobs``: log-softmax of the raw
  logits before temperature, penalties, or ``allowed_token_ids`` masks.
* llama.cpp returns pre-sampling probabilities unless ``post_sampling_probs``
  is set; this backend never sets it.

Both are pre-mask values, which is what the in-set softmax expects.

``HttpBackend`` posts the Jev request unchanged to ``{base}/v1/systemone``
(TypeSafe Jev or an OpenJev-style server).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Protocol

from ..utils.llm_usage import record_usage
from .models import SystemOneRequest, SystemOneResponse, Usage
from .readout import PromptFormat, build_prompt, readout_answer

PostFn = Callable[[dict[str, Any]], dict[str, Any]]


class SystemOneBackend(Protocol):
    def ask(self, request: SystemOneRequest) -> SystemOneResponse: ...


def _is_connection_reset(error: BaseException) -> bool:
    if isinstance(error, (ConnectionResetError, ConnectionAbortedError)):
        return True
    reason = getattr(error, "reason", None)
    return isinstance(reason, (ConnectionResetError, ConnectionAbortedError))


def _http_post(
    url: str,
    payload: dict[str, Any],
    api_key: str | None,
    timeout: float,
    *,
    reset_retries: int = 4,
) -> Any:
    """POST JSON. System One questions have no side effects, so a dropped
    connection (seen on Windows loopback) is retried a bounded number of times.
    """

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    for attempt in range(reset_retries + 1):
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"System One HTTP {error.code} from {url}: {detail}") from error
        except (OSError, urllib.error.URLError) as error:
            if attempt < reset_retries and _is_connection_reset(error):
                time.sleep(0.25 * (attempt + 1))
                continue
            raise


def parse_top_logprobs(response: dict[str, Any]) -> dict[str, float]:
    """Top-k logprobs of the first generated token.

    vLLM completions: ``logprobs.top_logprobs[0]`` is ``{token: logprob}``.
    llama.cpp: ``logprobs.content[0].top_logprobs`` is a list of
    ``{token, logprob}`` (chat-style).
    """

    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("completion response has no choices")
    logprobs = choices[0].get("logprobs") or {}
    top = logprobs.get("top_logprobs")
    if isinstance(top, list) and top and isinstance(top[0], dict) and "token" not in top[0]:
        return {str(k): float(v) for k, v in top[0].items()}
    content = logprobs.get("content")
    if isinstance(content, list) and content:
        items = content[0].get("top_logprobs") or []
        result: dict[str, float] = {}
        for item in items:
            token = str(item.get("token", ""))
            value = float(item["logprob"])
            if token not in result or value > result[token]:
                result[token] = value
        return result
    raise RuntimeError("completion response has no top logprobs; is logprobs enabled?")


class LocalReadoutBackend:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        top_k: int = 20,
        prompt_format: PromptFormat = "chatml",
        temperatures: dict[str, float] | None = None,
        timeout: float = 60.0,
        max_workers: int = 4,
        post: PostFn | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.top_k = top_k
        self.prompt_format = prompt_format
        self.temperatures = dict(temperatures or {})
        self.timeout = timeout
        self.max_workers = max_workers
        self._post = post or self._default_post

    def _default_post(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _http_post(f"{self.base_url}/completions", payload, self.api_key, self.timeout)

    def _payload(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": self.top_k,
            # llama.cpp keeps the shared state prefix in its KV cache.
            "cache_prompt": True,
        }

    def _ask_one(self, state: str, question) -> tuple[Any, int, int]:
        prompt = build_prompt(state, question, self.prompt_format)
        started = time.perf_counter()
        response = self._post(self._payload(prompt))
        latency_ms = (time.perf_counter() - started) * 1000
        record_usage(response, latency_ms, model=self.model)
        usage = response.get("usage") or {}
        answer = readout_answer(
            question,
            parse_top_logprobs(response),
            temperature=self.temperatures.get(question.type, 1.0),
        )
        return answer, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)

    def ask(self, request: SystemOneRequest) -> SystemOneResponse:
        items = list(request.questions.items())
        results: dict[str, tuple[Any, int, int]] = {}
        if items:
            # The first question warms the state prefix; the rest reuse it.
            first_name, first_q = items[0]
            results[first_name] = self._ask_one(request.state, first_q)
        rest = items[1:]
        if rest:
            workers = max(1, min(self.max_workers, len(rest)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    name: pool.submit(self._ask_one, request.state, question)
                    for name, question in rest
                }
                for name, future in futures.items():
                    results[name] = future.result()
        return SystemOneResponse(
            model=self.model,
            answers={name: results[name][0] for name, _ in items},
            usage=Usage(
                input_tokens=sum(r[1] for r in results.values()),
                output_tokens=sum(r[2] for r in results.values()),
            ),
        )


class HttpBackend:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        base = base_url.rstrip("/")
        self.url = base if base.endswith("/v1/systemone") else f"{base}/v1/systemone"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def ask(self, request: SystemOneRequest) -> SystemOneResponse:
        payload = request.model_dump(exclude_none=True)
        if "model" not in payload and self.model:
            payload["model"] = self.model
        data = _http_post(self.url, payload, self.api_key, self.timeout)
        return SystemOneResponse.model_validate(data)
