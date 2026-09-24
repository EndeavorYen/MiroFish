"""Measure local chat prefill, decode, prefix-cache, and VRAM."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def compute_throughput(
    prompt_tokens: int,
    completion_tokens: int,
    ttft_s: float,
    total_s: float,
) -> dict[str, float]:
    """Return prefill and decode tokens per second for one streamed call."""
    if ttft_s <= 0 or total_s <= ttft_s:
        raise ValueError("ttft_s must be > 0 and total_s must be > ttft_s")
    return {
        "prefill_tps": prompt_tokens / ttft_s,
        "decode_tps": (completion_tokens - 1) / (total_s - ttft_s),
    }


def prefix_cache_reduction(first_ttft_s: float, second_ttft_s: float) -> float:
    """Fraction by which the second TTFT dropped relative to the first."""
    return 1.0 - (second_ttft_s / first_ttft_s)


def parse_nvidia_smi_mib(output: str) -> int | None:
    """Parse the first ``memory.used`` MiB value from nvidia-smi csv output."""
    text = output.strip()
    if not text:
        return None
    line = text.splitlines()[0].strip()
    try:
        return int(line)
    except ValueError:
        return None


def build_repeated_prefix(seed_text: str, target_chars: int = 2000) -> str:
    """Repeat seed text until the prompt is about ``target_chars`` long."""
    seed = seed_text.strip()
    if not seed:
        raise ValueError("seed text is empty")
    parts: list[str] = []
    size = 0
    while size < target_chars:
        parts.append(seed)
        size += len(seed) + 1
    return "\n".join(parts)


def _post_stream(
    url: str,
    payload: dict[str, Any],
    api_key: str,
) -> tuple[float, float, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    ttft_s: float | None = None
    usage: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(request) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                delta = (choices[0].get("delta") or {}) if choices else {}
                text = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                if ttft_s is None and text:
                    ttft_s = time.perf_counter() - started
    except (ConnectionResetError, ConnectionAbortedError, TimeoutError) as exc:
        if ttft_s is None or not usage:
            raise urllib.error.URLError(str(exc)) from exc
    total_s = time.perf_counter() - started
    if ttft_s is None or total_s <= ttft_s or not usage:
        raise urllib.error.URLError("stream ended before a complete timed response")
    return ttft_s, total_s, usage


def _post_stream_once_retry(
    url: str,
    payload: dict[str, Any],
    api_key: str,
) -> tuple[float, float, dict[str, Any]]:
    """Retry one connection reset. A server that is down fails both attempts."""
    last_error: urllib.error.URLError | None = None
    for _ in range(2):
        try:
            return _post_stream(url, payload, api_key)
        except urllib.error.URLError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _chat_payload(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


class _VramSampler:
    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.peak_mib: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int | None:
        self._stop.set()
        self._thread.join(timeout=5)
        return self.peak_mib

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            except FileNotFoundError:
                return
            except (subprocess.TimeoutExpired, OSError):
                self._stop.wait(self.interval_s)
                continue
            if completed.returncode != 0:
                self._stop.wait(self.interval_s)
                continue
            used = parse_nvidia_smi_mib(completed.stdout)
            if used is not None:
                self.peak_mib = used if self.peak_mib is None else max(self.peak_mib, used)
            self._stop.wait(self.interval_s)


def _default_seed_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "golden_scenario"
        / "news_seed.txt"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark a local OpenAI-compatible LLM")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--output", default=None)
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", "local"))
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args(argv)

    seed = _default_seed_path().read_text(encoding="utf-8")
    prompt = build_repeated_prefix(seed, target_chars=2000)
    url = args.base_url.rstrip("/") + "/chat/completions"
    payload = _chat_payload(args.model, prompt, args.max_tokens)

    sampler = _VramSampler()
    sampler.start()
    try:
        first_ttft, first_total, first_usage = _post_stream_once_retry(url, payload, args.api_key)
        second_ttft, _second_total, _second_usage = _post_stream_once_retry(url, payload, args.api_key)

        success_count = 0
        failure_count = 0
        failure_errors: list[str] = []

        def _one() -> bool:
            _post_stream_once_retry(url, payload, args.api_key)
            return True

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(_one) for _ in range(args.concurrency)]
            for future in futures:
                try:
                    future.result()
                    success_count += 1
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError) as exc:
                    failure_count += 1
                    if len(failure_errors) < 3:
                        failure_errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        peak_vram_mib = sampler.stop()

    prompt_tokens = int(first_usage.get("prompt_tokens") or 0)
    completion_tokens = int(first_usage.get("completion_tokens") or 0)
    throughput = compute_throughput(
        prompt_tokens,
        completion_tokens,
        first_ttft,
        first_total,
    )
    report = {
        "concurrency": args.concurrency,
        "decode_tps": throughput["decode_tps"],
        "failure_count": failure_count,
        "failure_errors": failure_errors,
        "first_ttft_s": first_ttft,
        "max_model_len": args.max_model_len,
        "peak_vram_mib": peak_vram_mib,
        "prefill_tps": throughput["prefill_tps"],
        "prefix_cache_reduction": prefix_cache_reduction(first_ttft, second_ttft),
        "prompt_tokens": prompt_tokens,
        "second_ttft_s": second_ttft,
        "success_count": success_count,
    }
    rendered = json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
