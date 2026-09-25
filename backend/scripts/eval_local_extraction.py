"""Evaluate LocalExtractor on the golden seed against the hand-made gold graph.

Builds a LocalGraphStore from tests/fixtures/golden_scenario/news_seed.txt
(chunked like the app: TextProcessor, 500/50), then reports entity recall and
precision (normalized names, gold aliases count), type accuracy, edge recall
(unordered pair, and with type + direction), graph_build decode tokens from
llm_usage.jsonl, System One readout count, and VRAM delta during extraction.

Usage:
    uv run python scripts/eval_local_extraction.py --ner candidates
    uv run --with gliner --with torch --with jieba3 \
        python scripts/eval_local_extraction.py --ner gliner
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.graph.embedding import HashEmbedder, HttpEmbedder  # noqa: E402
from app.graph.local_extractor import LocalExtractor  # noqa: E402
from app.graph.local_store import LocalGraphStore, normalize_name  # noqa: E402
from app.graph.store import TextEpisode  # noqa: E402
from app.services.text_processor import TextProcessor  # noqa: E402
from app.system_one.backends import LocalReadoutBackend  # noqa: E402
from app.system_one.client import SystemOneClient, load_temperatures  # noqa: E402
from app.utils.llm_usage import usage_stage  # noqa: E402

FIXTURES = BACKEND_DIR / "tests" / "fixtures" / "golden_scenario"
HOLDOUT = BACKEND_DIR / "tests" / "fixtures" / "holdout_extraction"
G3_THRESHOLD = 0.80


class VramSampler:
    """Sample nvidia-smi memory.used (MiB) in a background thread."""

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def read() -> int | None:
        if not shutil.which("nvidia-smi"):
            return None
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        return int(out.splitlines()[0]) if out else None

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self.read()
            if value is not None:
                self.samples.append(value)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


class CountingClient(SystemOneClient):
    def __init__(self, backend) -> None:
        super().__init__(backend)
        self.questions = 0

    def ask(self, request):
        self.questions += len(request.questions)
        return super().ask(request)


def evaluate(store: LocalGraphStore, graph_id: str, gold: dict) -> dict:
    group_of = {}
    type_of = {}
    for entity in gold["entities"]:
        for name in [entity["name"], *entity["aliases"]]:
            group_of[normalize_name(name)] = entity["name"]
        type_of[entity["name"]] = entity["type"]

    nodes = store.list_nodes(graph_id)
    node_group = {}
    for node in nodes:
        names = [node.name, *node.attributes.get("aliases", [])]
        groups = {group_of.get(normalize_name(n)) for n in names} - {None}
        node_group[node.uuid] = next(iter(groups)) if groups else None
    matched = {g for g in node_group.values() if g}
    typed_ok = sum(
        1
        for node in nodes
        if node_group[node.uuid] and type_of[node_group[node.uuid]] in node.labels
    )
    edges = store.list_edges(graph_id)
    pairs = set()
    typed = set()
    for edge in edges:
        s, t = node_group.get(edge.source_node_uuid), node_group.get(edge.target_node_uuid)
        if s and t:
            pairs.add(frozenset((s, t)))
            typed.add((s, edge.name, t))
    gold_rel = gold["relations"]
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "entity_recall": round(len(matched) / len(gold["entities"]), 4),
        "entity_precision": round(
            sum(1 for g in node_group.values() if g) / max(len(nodes), 1), 4
        ),
        "type_accuracy_on_matched": round(
            typed_ok / max(sum(1 for g in node_group.values() if g), 1), 4
        ),
        "edge_recall_pair": round(
            sum(1 for r in gold_rel if frozenset((r["source"], r["target"])) in pairs)
            / len(gold_rel),
            4,
        ),
        "edge_recall_typed": round(
            sum(1 for r in gold_rel if (r["source"], r["type"], r["target"]) in typed)
            / len(gold_rel),
            4,
        ),
        "missed_entities": sorted({e["name"] for e in gold["entities"]} - matched),
        "unmatched_nodes": sorted(n.name for n in nodes if not node_group[n.uuid]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ner", choices=["candidates", "gliner"], default="candidates")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen3.5-4b")
    parser.add_argument("--embedder", choices=["http", "hash"], default="http")
    parser.add_argument("--embed-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--embed-model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument(
        "--fixture", type=Path, default=FIXTURES,
        help=f"directory with news_seed.txt, ontology.json, gold_graph.json (held-out: {HOLDOUT})",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    fixture = args.fixture

    ontology = json.loads((fixture / "ontology.json").read_text(encoding="utf-8"))
    gold = json.loads((fixture / "gold_graph.json").read_text(encoding="utf-8"))
    text = (fixture / "news_seed.txt").read_text(encoding="utf-8")

    embedder = (
        HashEmbedder()
        if args.embedder == "hash"
        else HttpEmbedder(
            base_url=args.embed_base_url,
            model=args.embed_model,
            query_prefix="query: " if "e5" in args.embed_model else "",
            passage_prefix="passage: " if "e5" in args.embed_model else "",
        )
    )
    client = CountingClient(
        LocalReadoutBackend(
            base_url=args.base_url, model=args.model, temperatures=load_temperatures()
        )
    )
    extractor = LocalExtractor(client, embedder=embedder, ner=args.ner)
    if args.ner == "gliner":
        extractor._load_gliner()  # load before measuring VRAM / time

    work = Path(tempfile.mkdtemp(prefix="local_extract_eval_"))
    metrics_dir = work / "metrics"
    store = LocalGraphStore(str(work / "graphs"), embedder=embedder, extractor=extractor)
    graph_id = store.create_graph("golden", graph_id="golden_eval")
    store.set_ontology(graph_id, ontology)
    chunks = TextProcessor.split_text(text, chunk_size=args.chunk_size, overlap=args.overlap)

    vram_before = VramSampler.read()
    started = time.perf_counter()
    with usage_stage("graph_build", metrics_dir=str(metrics_dir)), VramSampler() as vram:
        store.add_text_episodes(graph_id, [TextEpisode(c) for c in chunks], durable=True)
    elapsed = time.perf_counter() - started

    usage_path = metrics_dir / "llm_usage.jsonl"
    rows = [json.loads(line) for line in usage_path.read_text(encoding="utf-8").splitlines()] if usage_path.exists() else []
    report = {
        "fixture": fixture.name,
        "ner": args.ner,
        "model": args.model,
        "embedder": args.embedder if args.embedder == "hash" else args.embed_model,
        "chunks": len(chunks),
        "seconds": round(elapsed, 1),
        "system_one_questions": client.questions,
        "llm_usage": {
            "rows": len(rows),
            "stages": sorted({r["stage"] for r in rows}),
            "graph_build_prompt_tokens": sum(r["prompt_tokens"] for r in rows if r["stage"] == "graph_build"),
            "graph_build_decode_tokens": sum(r["completion_tokens"] for r in rows if r["stage"] == "graph_build"),
        },
        "vram_mib": {
            "before": vram_before,
            "peak_during": max(vram.samples) if vram.samples else None,
            "delta": (max(vram.samples) - vram_before) if vram.samples and vram_before is not None else None,
        },
        **evaluate(store, graph_id, gold),
    }
    report["gate_g3"] = {
        "threshold": G3_THRESHOLD,
        "entity_recall": report["entity_recall"],
        "passed": report["entity_recall"] >= G3_THRESHOLD,
        "reference": "hand-made gold graph (no Zep Cloud call)",
    }
    store.close()
    text_out = json.dumps(report, ensure_ascii=False, indent=2)
    print(text_out)
    if args.out:
        args.out.write_text(text_out + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
