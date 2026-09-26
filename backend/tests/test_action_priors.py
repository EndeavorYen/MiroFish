import json
import random
from collections import Counter

import pytest

import scripts.fit_action_priors as fit
from app.simulation_policy.priors import load_action_priors, with_priors
from app.simulation_policy.taxonomy import load_taxonomy
from app.system_one.models import ChoiceAnswer, SystemOneResponse
from app.system_one.tree import apply_priors, ask_tree


def test_apply_priors_is_a_product_of_experts():
    p = {"create": 0.7, "engage": 0.1, "none": 0.2}
    out = apply_priors(p, {"engage": 7.0})
    assert out["engage"] == pytest.approx(0.7 / 1.6)
    assert out["create"] == pytest.approx(out["engage"])
    assert apply_priors(p, None) == p
    assert apply_priors(p, {"create": 0, "engage": 0, "none": 0}) == p  # all vanished: unchanged


class FixedClient:
    def __init__(self, probs):
        self.probs = probs

    def ask(self, request):
        (name,) = request.questions
        probs = dict(self.probs)
        return SystemOneResponse(answers={name: ChoiceAnswer(choice=max(probs, key=probs.get), probabilities=probs, confidence=max(probs.values()))})


def test_ask_tree_samples_from_weighted_distribution():
    tree = {"name": "action", "question": {"type": "choice", "instructions": "?", "criteria": {"a": "A", "b": "B"}},
            "priors": {"b": 9.0}}
    client = FixedClient({"a": 0.5, "b": 0.5})
    picks = Counter(ask_tree(client, "s", tree, random.Random(i))[0].choice for i in range(2000))
    assert 0.85 < picks["b"] / 2000 < 0.95
    step = ask_tree(client, "s", tree, random.Random(0))[0]
    assert step.readout == {"a": 0.5, "b": 0.5} and step.probabilities["b"] == pytest.approx(0.9)


def test_fit_node_matches_target_mix_and_keeps_activity():
    readouts = [{"create": 0.7, "engage": 0.05, "none": 0.25}, {"create": 0.6, "engage": 0.1, "none": 0.3}]
    w = fit.fit_node(["engage", "create", "none"], Counter({"engage": 60, "create": 40}), readouts)
    assert w["none"] == 1.0
    adjusted = [apply_priors(r, w) for r in readouts]
    mean = {k: sum(a[k] for a in adjusted) / 2 for k in ("engage", "create", "none")}
    active = mean["engage"] + mean["create"]
    assert mean["engage"] / active == pytest.approx(60.5 / 101, abs=0.01)
    assert active == pytest.approx(0.725, abs=0.01)  # original non-exit mass


def test_fit_platform_maps_actions_to_tree_paths():
    taxonomy = load_taxonomy("twitter")
    decisions = [{"path": ["create", "create_post"], "probs": [{"engage": 0.1, "create": 0.8, "social": 0.05, "none": 0.05},
                                                             {"create_post": 0.9, "none": 0.1}], "platform": "twitter"}]
    priors = fit.fit_platform(Counter({"LIKE_POST": 50, "CREATE_POST": 25}), decisions, taxonomy)
    assert priors[""]["engage"] > priors[""]["create"]
    assert "engage" in priors  # fitted from a uniform readout when no B agent reached it
    assert priors["engage"]["like_post"] > priors["engage"]["quote_post"]


def test_load_and_attach_priors(tmp_path, monkeypatch):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"priors": {"twitter": {"": {"engage": 5.0}, "engage": {"like_post": 3.0}}}}), encoding="utf-8")
    priors = load_action_priors(str(path))
    taxonomy = with_priors(load_taxonomy("twitter"), priors["twitter"])
    assert taxonomy.tree["priors"] == {"engage": 5.0}
    assert taxonomy.tree["children"]["engage"]["priors"] == {"like_post": 3.0}
    assert "priors" not in load_taxonomy("twitter").tree  # cached original untouched
    assert load_action_priors("off") == {}
    with pytest.raises(FileNotFoundError):
        load_action_priors(str(tmp_path / "missing.json"))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"priors": {"twitter": {"": {"nope": 1.0}}}}), encoding="utf-8")
    with pytest.raises(ValueError):
        with_priors(load_taxonomy("twitter"), load_action_priors(str(bad))["twitter"])
