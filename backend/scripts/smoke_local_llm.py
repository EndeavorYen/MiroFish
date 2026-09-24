"""Record ontology and profile generation failures. Does not retry or fix them."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable


def _record(results: list[dict[str, Any]], step: str, action: Callable[[], None]) -> None:
    try:
        action()
    except Exception as exc:
        results.append(
            {
                "step": step,
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        return
    results.append(
        {
            "step": step,
            "ok": True,
            "error_type": None,
            "error_message": None,
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke ontology and profile generation and record JSON failures"
    )
    parser.parse_args(argv)

    backend_dir = Path(__file__).resolve().parents[1]
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    # Config.load_dotenv(override=True) would replace an explicit shell env.
    # Keep LLM_* values that were already set when this process started.
    preset = {
        key: os.environ[key]
        for key in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_NAME")
        if os.environ.get(key)
    }
    from app.config import Config

    for key, value in preset.items():
        os.environ[key] = value
        setattr(Config, key, value)
    from app.services.oasis_profile_generator import OasisProfileGenerator
    from app.services.ontology_generator import OntologyGenerator
    from app.services.zep_entity_reader import EntityNode

    fixture_dir = backend_dir / "tests" / "fixtures" / "golden_scenario"
    news = (fixture_dir / "news_seed.txt").read_text(encoding="utf-8")
    requirement = (fixture_dir / "simulation_requirement.txt").read_text(encoding="utf-8")
    results: list[dict[str, Any]] = []

    def run_ontology() -> None:
        OntologyGenerator().generate([news], requirement)

    _record(results, "ontology", run_ontology)

    entities = [
        EntityNode(
            uuid="smoke-1",
            name="市民甲",
            labels=["Person"],
            summary="東海市居民",
            attributes={},
        ),
        EntityNode(
            uuid="smoke-2",
            name="東海市府",
            labels=["Organization"],
            summary="東海市政府",
            attributes={},
        ),
        EntityNode(
            uuid="smoke-3",
            name="地方媒體",
            labels=["MediaOutlet"],
            summary="報導市政的媒體",
            attributes={},
        ),
    ]
    generator = OasisProfileGenerator(zep_api_key="")
    for index, entity in enumerate(entities):
        _record(
            results,
            f"profile:{entity.name}",
            lambda entity=entity, index=index: generator.generate_profile_from_entity(
                entity,
                user_id=index,
                use_llm=True,
            ),
        )

    payload = {
        "model": Config.LLM_MODEL_NAME or "",
        "base_url": Config.LLM_BASE_URL or "",
        "results": results,
    }
    print(json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
