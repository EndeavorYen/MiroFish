"""Text embedders for the local graph store.

``HttpEmbedder`` calls an OpenAI-compatible ``/v1/embeddings`` endpoint (the
CPU ``local-embed`` service from the local compose profile, llama.cpp, or
Ollama). ``HashEmbedder`` is a deterministic character n-gram hashing
embedder with no model; tests and offline benchmarks use it.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from typing import Protocol, Sequence

from .text_index import char_ngrams


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


class HashEmbedder:
    """Signed feature hashing of character 1-3 grams, L2-normalised."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for gram in char_ngrams(text):
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.dim
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[index] += sign
        return _normalize(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class HttpEmbedder:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        query_prefix: str = "",
        passage_prefix: str = "",
        batch_size: int = 32,
        timeout: float = 20.0,
        max_input_chars: int = 400,
    ) -> None:
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model = model
        self.api_key = api_key
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self.timeout = timeout
        # Small encoders (e5-small: 512 tokens) reject longer inputs; CJK is
        # roughly one token per character, so cut well below the limit.
        self.max_input_chars = max_input_chars

    def _post(self, inputs: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = json.dumps({"model": self.model, "input": inputs}).encode("utf-8")
        # Embedding is idempotent: retry dropped or stalled loopback
        # connections a bounded number of times.
        retryable = (ConnectionResetError, ConnectionAbortedError, TimeoutError)
        for attempt in range(3):
            request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError:
                raise
            except (OSError, urllib.error.URLError) as error:
                reason = getattr(error, "reason", error)
                if attempt == 2 or not isinstance(reason, retryable):
                    raise
                time.sleep(0.25 * (attempt + 1))
        rows = sorted(data["data"], key=lambda item: item["index"])
        return [_normalize([float(v) for v in row["embedding"]]) for row in rows]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        items = [self.passage_prefix + text[: self.max_input_chars] for text in texts]
        for start in range(0, len(items), self.batch_size):
            vectors.extend(self._post(items[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._post([self.query_prefix + text[: self.max_input_chars]])[0]


def make_embedder() -> Embedder:
    from ..config import Config

    kind = Config.GRAPH_EMBEDDER
    if kind == "hash":
        return HashEmbedder()
    if kind == "http":
        model = Config.EMBED_MODEL_NAME
        # e5 models are trained with "query: " / "passage: " prefixes.
        is_e5 = "e5" in model.lower()
        return HttpEmbedder(
            base_url=Config.EMBED_BASE_URL,
            model=model,
            api_key=Config.EMBED_API_KEY,
            query_prefix=Config.EMBED_QUERY_PREFIX
            if Config.EMBED_QUERY_PREFIX is not None
            else ("query: " if is_e5 else ""),
            passage_prefix=Config.EMBED_PASSAGE_PREFIX
            if Config.EMBED_PASSAGE_PREFIX is not None
            else ("passage: " if is_e5 else ""),
        )
    raise ValueError(f"GRAPH_EMBEDDER must be http or hash, got {kind!r}")
