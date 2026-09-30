"""A small OpenAI-compatible embedding server (``POST /v1/embeddings``).

For hosts without llama.cpp (e.g. a Colab runtime): serves
multilingual-e5-small with mean pooling and L2 normalisation, the same as
``llama-server --embedding --pooling mean``, so ``EMBED_BASE_URL`` can point
at it unchanged.

Usage:
    uv run python scripts/embed_server.py [--model intfloat/multilingual-e5-small] [--port 8001]
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def build_encoder(model_name: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    lock = threading.Lock()

    @torch.no_grad()
    def encode(texts: list[str]) -> list[list[float]]:
        with lock:
            batch = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return torch.nn.functional.normalize(pooled, dim=-1).cpu().tolist()

    return encode


def make_handler(encode, model_name: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") in ("/health", "/v1/models"):
                self._send(200, {"status": "ok", "data": [{"id": model_name, "object": "model"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/v1/embeddings":
                self._send(404, {"error": "not found"})
                return
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            texts = request.get("input", [])
            texts = [texts] if isinstance(texts, str) else [str(t) for t in texts]
            vectors = encode(texts) if texts else []
            self._send(200, {
                "object": "list",
                "model": request.get("model", model_name),
                "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)],
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            })

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(build_encoder(args.model), args.model))
    print(f"embedding server on http://{args.host}:{args.port}/v1 ({args.model})", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
