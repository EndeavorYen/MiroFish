"""Benchmark LocalGraphStore.search on synthetic data (default 10k edges).

Latency is measured end to end per call, including the query embedding.

Usage:
    uv run python scripts/bench_local_graph_search.py --embedder hash
    uv run python scripts/bench_local_graph_search.py --embedder http \
        --embed-base-url http://127.0.0.1:8001/v1 \
        --embed-model intfloat/multilingual-e5-small
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.graph.embedding import HashEmbedder, HttpEmbedder  # noqa: E402
from app.graph.extractor import ExtractedEntity, ExtractedRelation, Extraction  # noqa: E402
from app.graph.local_store import LocalGraphStore  # noqa: E402
from app.graph.store import TextEpisode  # noqa: E402

SURNAMES = "陳林黃張李王吳劉蔡楊許鄭謝洪郭邱曾廖賴徐周葉蘇莊呂江何蕭羅高潘簡朱鍾彭游詹胡施沈余趙盧梁顏柯孫魏翁戴范宋方鄧杜傅侯曹薛丁卓馬阮董唐溫藍蔣石古紀姚連馮歐程湯黃田康姜汪白鄒尤巫鐘黎涂龔嚴韓袁金童陸夏柳凃邵錢伍倪溫于譚駱熊任甘秦顧毛章史官萬俞雷粘饒張闕凌崔尹孔辛歐陽"
GIVEN = "志明雅婷建宏淑芬俊傑怡君家豪美玲冠宇佳穎承恩欣怡宗翰惠雯柏翰詩涵子軒心怡"
ORG_PREFIX = ["東海", "濱海", "高新", "凌雲", "天際", "海港", "北辰", "南嶺", "星河", "雲端"]
ORG_SUFFIX = ["科技公司", "運輸公司", "市政府", "工會", "日報", "大學", "銀行", "協會", "醫院", "研究院"]
RELATIONS = [
    ("SUPPORTS", "{a}公開表示支持{b}的低空試點方案"),
    ("OPPOSES", "{a}在記者會上反對{b}提出的航線規劃"),
    ("WORKS_FOR", "{a}目前在{b}擔任主管"),
    ("REPORTS_ON", "{a}報導了{b}的最新營運數據"),
    ("CRITICIZES", "{a}批評{b}對噪音問題的回應太慢"),
    ("PARTNERS_WITH", "{a}宣布與{b}合作推動空中計程車"),
    ("REGULATES", "{a}要求{b}提交安全評估報告"),
    ("MENTIONS", "{a}在貼文中提到{b}的票價調整"),
]


def build_entities(rng: random.Random, n_people: int, n_orgs: int) -> list[tuple[str, str]]:
    people = set()
    while len(people) < n_people:
        people.add(rng.choice(SURNAMES) + rng.choice(GIVEN[0::2]) + rng.choice(GIVEN[1::2]))
    orgs = set()
    while len(orgs) < n_orgs:
        orgs.add(rng.choice(ORG_PREFIX) + rng.choice(ORG_PREFIX) + rng.choice(ORG_SUFFIX))
    return [(p, "Person") for p in sorted(people)] + [(o, "Organization") for o in sorted(orgs)]


class ScriptedExtractor:
    """Return pre-built extractions in order (bypasses real extraction)."""

    def __init__(self, extractions: list[Extraction]) -> None:
        self._items = iter(extractions)

    def extract(self, text, ontology, known_entities=None):
        return next(self._items)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", type=int, default=10_000)
    parser.add_argument("--people", type=int, default=1500)
    parser.add_argument("--orgs", type=int, default=500)
    parser.add_argument("--per-episode", type=int, default=20)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--embedder", choices=["hash", "http"], default="hash")
    parser.add_argument("--embed-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--embed-model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    entities = build_entities(rng, args.people, args.orgs)
    by_name = dict(entities)
    extractions, texts = [], []
    seen = set()
    remaining = args.edges
    while remaining > 0:
        relations, used = [], {}
        while len(relations) < min(args.per_episode, remaining):
            a, b = rng.sample(entities, 2)
            name, template = rng.choice(RELATIONS)
            fact = template.format(a=a[0], b=b[0]) + f"（第{rng.randint(1, 24)}回合）"
            if fact in seen:
                continue
            seen.add(fact)
            relations.append(ExtractedRelation(name=name, source=a[0], target=b[0], fact=fact))
            used[a[0]] = fact
            used[b[0]] = fact
        extractions.append(
            Extraction(
                entities=[ExtractedEntity(n, by_name[n], summary=f) for n, f in used.items()],
                relations=relations,
            )
        )
        texts.append("\n".join(r.fact for r in relations))
        remaining -= len(relations)

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
    data_dir = args.data_dir or Path(tempfile.mkdtemp(prefix="local_graph_bench_"))
    store = LocalGraphStore(str(data_dir), embedder=embedder, extractor=ScriptedExtractor(extractions))
    graph_id = store.create_graph("bench", graph_id=f"bench_{args.seed}_{args.edges}")

    started = time.perf_counter()
    store.add_text_episodes(graph_id, [TextEpisode(t) for t in texts], durable=True)
    ingest_s = time.perf_counter() - started
    node_count = len(store.list_nodes(graph_id))
    edge_count = len(store.list_edges(graph_id))

    query_templates = [
        "{n}支持什麼", "誰反對{n}", "{n}在哪裡工作", "{n}的票價", "{n}和誰合作", "噪音問題 {n}",
    ]
    queries = [
        rng.choice(query_templates).format(n=rng.choice(entities)[0]) for _ in range(args.queries)
    ]
    # Warm-up: first call opens caches and loads the vec index pages.
    store.search(graph_id, queries[0], "edges", args.limit)
    latencies = {"edges": [], "nodes": []}
    hits = 0
    for query in queries:
        for scope in ("edges", "nodes"):
            t0 = time.perf_counter()
            result = store.search(graph_id, query, scope, args.limit, ranking="fusion")
            latencies[scope].append((time.perf_counter() - t0) * 1000)
            if scope == "edges" and result.edges:
                name = next(n for n, _ in entities if n in query)
                hits += any(name in e.fact for e in result.edges)

    def pct(values, q):
        ordered = sorted(values)
        return round(ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))], 2)

    report = {
        "embedder": args.embedder if args.embedder == "hash" else args.embed_model,
        "nodes": node_count,
        "edges": edge_count,
        "ingest_seconds": round(ingest_s, 1),
        "queries": len(queries),
        "limit": args.limit,
        "latency_ms": {
            scope: {
                "p50": pct(values, 0.50),
                "p95": pct(values, 0.95),
                "max": round(max(values), 2),
                "mean": round(statistics.mean(values), 2),
            }
            for scope, values in latencies.items()
        },
        "edge_query_entity_hit_rate": round(hits / len(queries), 3),
        "db_bytes": (data_dir / f"{graph_id}.sqlite").stat().st_size,
    }
    store.close()
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
