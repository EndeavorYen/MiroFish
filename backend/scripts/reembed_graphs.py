"""Recompute local graph vectors with the current embedder (#61).

A graph records which embedder wrote its vectors; one built by another
embedder (for example the mis-converted e5-small GGUF) is searched by
keywords only until it is re-embedded. Stop the backend first: this
rewrites every vector of each graph.

    uv run python scripts/reembed_graphs.py            # every graph
    uv run python scripts/reembed_graphs.py --check    # report only
    uv run python scripts/reembed_graphs.py mirofish_0c3bec138b094bf4
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app.config import Config  # noqa: E402  (applies MIROFISH_PROFILE)
from app.graph.embedding import make_embedder  # noqa: E402
from app.graph.extractor import StubExtractor  # noqa: E402
from app.graph.local_store import LocalGraphStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("graph_ids", nargs="*", help="default: every graph")
    parser.add_argument("--data-dir", default=Config.GRAPH_DATA_DIR)
    parser.add_argument("--check", action="store_true", help="report which graphs need it; change nothing")
    args = parser.parse_args()

    store = LocalGraphStore(args.data_dir, embedder=make_embedder(), extractor=StubExtractor({}))
    try:
        for graph_id in args.graph_ids or store.graph_ids():
            if args.check:
                # "no fingerprint": built before #61, maybe by the broken GGUF.
                print(f"{graph_id}\t{store.embedding_state(graph_id)}")
                continue
            started = time.perf_counter()
            counts = store.reembed(graph_id)
            print(f"{graph_id}\t{counts['nodes']} nodes, {counts['edges']} edges\t"
                  f"{time.perf_counter() - started:.1f} s")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
