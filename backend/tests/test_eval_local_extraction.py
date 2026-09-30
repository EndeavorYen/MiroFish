"""The extraction evaluation reports split entities and wrong boundaries (#64)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_local_extraction.py"
spec = importlib.util.spec_from_file_location("eval_local_extraction", SCRIPT)
eval_local_extraction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_local_extraction)


class FakeStore:
    def __init__(self, names):
        self.nodes = [SimpleNamespace(uuid=str(i), name=n, labels=["Entity", "Company"], attributes={})
                      for i, n in enumerate(names)]

    def list_nodes(self, graph_id):
        return self.nodes

    def list_edges(self, graph_id):
        return []


GOLD = {
    "entities": [
        {"name": "雲梯科技", "type": "Company", "aliases": ["雲梯"]},
        {"name": "林書瑤", "type": "Company", "aliases": []},
    ],
    "relations": [{"source": "林書瑤", "target": "雲梯科技", "type": "LEADS"}],
}


def test_split_entities_and_wrong_boundaries_are_reported():
    report = eval_local_extraction.evaluate(FakeStore(["雲梯科技", "雲梯", "師林書瑤", "無關"]), "g", GOLD)
    assert report["duplicate_groups"] == ["雲梯科技"]
    assert report["boundary_errors"] == ["師林書瑤"]
    assert report["missed_entities"] == ["林書瑤"]
