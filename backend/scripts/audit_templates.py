"""Audit locale templates against the post-stance readout (#45, #52).

Every ``simContent`` template line is filled with the entity and topic of
three calibration scenarios and scored with the zero-decode stance check the
content tier uses (``system_one_stance_score``). A line passes when at least
two of the three fills land on its own level. Evaluation scenarios are not
used, so the audit cannot tune templates to them.

The first audit (2026-09-28) found 46/90 zh lines on their level: "pos" lines
that name support outright read as strong support (13/18) and "neg_strong"
lines read as plain opposition (10/18).

Usage:
    uv run python scripts/audit_templates.py --lang zh [--out audit.json] [--candidates lines.txt --band pos]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "scripts"))

LEVELS = ("neg_strong", "neg", "neu", "pos", "pos_strong")
FILLS = {
    "zh": [("北港市自來水事業處", "水費調漲方案"), ("西平縣政府教育處", "小校併校計畫"), ("海峰綠能", "離岸風場開發案")],
    "en": [("Ashford City Council", "the bridge toll"), ("Lakemont Town Council", "the library rebuild"),
           ("the county board", "the new bus plan")],
}


def audit_line(line: str, band: str, fills: list[tuple[str, str]], level_of: Callable[[str, str], str]) -> dict:
    """``level_of(text, topic)``: the level of a filled line, asked about that topic."""

    got = [level_of(line.format(entity=entity, topic=topic, target=""), topic) for entity, topic in fills]
    return {"line": line, "band": band, "got": got, "passed": sum(g == band for g in got) >= 2}


def audit(templates: dict, fills: list[tuple[str, str]], level_of: Callable[[str, str], str]) -> dict:
    rows = []
    for kind, bands in templates.items():
        for band, lines in bands.items():
            for line in lines:
                rows.append({"kind": kind, **audit_line(line, band, fills, level_of)})
    confusion = Counter((row["band"], got) for row in rows for got in row["got"])
    return {
        "passed": sum(row["passed"] for row in rows),
        "total": len(rows),
        "confusion": {band: {got: confusion[(band, got)] for got in LEVELS if confusion[(band, got)]} for band in LEVELS},
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", choices=["zh", "en"], default="zh")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--candidates", type=Path, help="one candidate line per row, all for --band")
    parser.add_argument("--band", choices=LEVELS)
    args = parser.parse_args(argv)

    import golden_pipeline as gp

    import tempfile

    # Metrics and graph data go to a temp dir, never into the repo.
    gp.force_local_config(Path(tempfile.mkdtemp(prefix="mirofish-script-")))
    from app.simulation_policy.tiers import (
        level_index, load_templates, stance_check_question, system_one_stance_score,
    )

    def level_of(text: str, topic: str) -> str:
        # The same event-conditioned question the content check asks.
        question = stance_check_question(topic, args.lang)
        return LEVELS[level_index(system_one_stance_score(text, question=question))]

    fills = FILLS[args.lang]
    if args.candidates:
        for line in args.candidates.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = audit_line(line.strip(), args.band, fills, level_of)
                print("PASS" if row["passed"] else "fail", " ".join(row["got"]), "|", row["line"])
        return 0
    report = audit(load_templates(args.lang)["templates"], fills, level_of)
    print(f"{report['passed']}/{report['total']} lines on their level")
    for band, row in report["confusion"].items():
        print(f"  {band:11} -> {row}")
    for row in report["rows"]:
        if not row["passed"]:
            print(f"  {row['kind']}/{row['band']} read {row['got']}: {row['line']}")
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
