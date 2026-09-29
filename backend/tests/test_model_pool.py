import json

import pytest

from app.system_one.models import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
)

POOL = [
    {"name": "qwen", "base_url": "http://127.0.0.1:8000/v1", "model": "qwen3.5-4b",
     "roles": ["readout", "generate", "decide"]},
    {"name": "gemma", "base_url": "http://127.0.0.1:8002/v1", "model": "gemma-3-4b",
     "roles": ["readout", "generate"], "prompt_format": "plain"},
]


def test_default_pool_is_the_single_configured_model(monkeypatch):
    from app.model_pool import load_pool

    monkeypatch.delenv("MODEL_POOL", raising=False)
    pool = load_pool()
    assert len(pool) == 1
    assert pool[0].roles == frozenset({"readout", "generate", "decide"})


def test_pool_from_env_and_roles(monkeypatch):
    from app.model_pool import load_pool, models_for

    monkeypatch.setenv("MODEL_POOL", json.dumps(POOL))
    pool = load_pool()
    assert [m.name for m in models_for(pool, "generate")] == ["qwen", "gemma"]
    assert [m.name for m in models_for(pool, "decide")] == ["qwen"]
    assert pool[1].prompt_format == "plain"


def test_assignment_is_stable_and_uses_every_model(monkeypatch):
    from app.model_pool import assign, load_pool, models_for

    monkeypatch.setenv("MODEL_POOL", json.dumps(POOL))
    generate = models_for(load_pool(), "generate")
    first = [assign(ref, generate).name for ref in range(40)]
    assert first == [assign(ref, generate).name for ref in range(40)]
    assert set(first) == {"qwen", "gemma"}


class Fake:
    def __init__(self, probs, fail=False):
        self.probs = probs
        self.fail = fail
        self.calls = []

    def ask(self, request):
        self.calls.append(set(request.questions))
        if self.fail:
            raise RuntimeError("endpoint down")
        answers = {}
        for name, q in request.questions.items():
            if isinstance(q, ChoiceQuestion):
                best = max(self.probs, key=self.probs.get)
                answers[name] = ChoiceAnswer(choice=best, probabilities=dict(self.probs), confidence=self.probs[best])
            elif isinstance(q, ScoreQuestion):
                p = [self.probs.get(level, 0.0) for level in q.criteria]
                answers[name] = ScoreAnswer(score=sum(i * x for i, x in enumerate(p)),
                                            probabilities=dict(zip(q.criteria, p)), confidence=max(p))
            else:
                answers[name] = NoulAnswer(noul=self.probs.get("yes", 0.5))
        return SystemOneResponse(answers=answers)


def _choice_request():
    return SystemOneRequest(state="s", questions={
        "stance": ChoiceQuestion(instructions="?", criteria={"a": "A", "b": "B"}),
        "other": ChoiceQuestion(instructions="?", criteria={"a": "A", "b": "B"}),
    })


def test_ensemble_averages_scoped_questions_only():
    from app.model_pool import EnsembleBackend

    one, two = Fake({"a": 0.8, "b": 0.2}), Fake({"a": 0.2, "b": 0.8})
    backend = EnsembleBackend([one, two], scope={"stance"})
    answers = backend.ask(_choice_request()).answers
    assert answers["stance"].probabilities == pytest.approx({"a": 0.5, "b": 0.5})
    assert answers["other"].probabilities == pytest.approx({"a": 0.8, "b": 0.2})
    assert two.calls == [{"stance"}]


def test_ensemble_score_and_noul_are_recomputed():
    from app.model_pool import EnsembleBackend

    levels = ["lo", "mid", "hi"]
    one, two = Fake({"lo": 1.0, "yes": 0.2}), Fake({"hi": 1.0, "yes": 0.6})
    request = SystemOneRequest(state="s", questions={
        "s": ScoreQuestion(instructions="?", criteria=levels), "n": NoulQuestion(instructions="?"),
    })
    answers = EnsembleBackend([one, two], scope=None).ask(request).answers
    assert answers["s"].score == pytest.approx(1.0)
    assert answers["n"].noul == pytest.approx(0.4)


def test_ensemble_degrades_when_one_endpoint_fails():
    from app.model_pool import EnsembleBackend

    good, bad = Fake({"a": 0.9, "b": 0.1}), Fake({}, fail=True)
    answers = EnsembleBackend([good, bad], scope=None).ask(_choice_request()).answers
    assert answers["stance"].probabilities == pytest.approx({"a": 0.9, "b": 0.1})
    with pytest.raises(RuntimeError):
        EnsembleBackend([Fake({}, fail=True)], scope=None).ask(_choice_request())


def test_pooled_generation_falls_back_to_the_first_model():
    from app.model_pool import PooledGenerator

    calls = []

    def make(name, fail=False):
        def fn(prompt, max_tokens):
            calls.append(name)
            if fail:
                raise RuntimeError("down")
            return f"{name} text", 3
        return fn

    gen = PooledGenerator([("qwen", make("qwen")), ("gemma", make("gemma", fail=True))])
    refs = [ref for ref in range(40) if gen.name_for(ref) == "gemma"]
    text, tokens = gen.for_persona(refs[0])("p", 10)
    assert (text, tokens) == ("qwen text", 3)
    assert calls[-2:] == ["gemma", "qwen"]


def test_provider_records_the_generating_model(tmp_path):
    from app.model_pool import PooledGenerator
    from app.simulation_policy.content import ContentIntent
    from app.simulation_policy.tiers import TieredContentProvider, load_templates

    gen = PooledGenerator([("qwen", lambda p, n: ("q", 2)), ("gemma", lambda p, n: ("g", 2))])
    provider = TieredContentProvider(
        templates=load_templates("zh"), llm_fn=gen.default, generator=gen, entities=["X"],
        followers={i: 100 - i for i in range(10)}, top_k_percent=100, budget_per_round=10_000,
        metrics_path=str(tmp_path / "m.jsonl"),
    )
    for ref in range(6):
        provider.generate(ContentIntent(kind="opinion", stance=0.5, intensity=0.5, target_ref=None,
                                        persona_ref=ref, platform="twitter", round_num=1))
    provider.flush()
    row = json.loads((tmp_path / "m.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert sum(row["models"].values()) == 6
    assert set(row["models"]) <= {"qwen", "gemma"}


def test_build_tiered_provider_uses_the_pool(monkeypatch, tmp_path):
    from app.simulation_policy.tiers import build_tiered_provider

    monkeypatch.setenv("MODEL_POOL", json.dumps(POOL))
    provider = build_tiered_provider("twitter", str(tmp_path), {"agent_configs": []}, score_fn=lambda t: 0.5)
    assert [name for name, _ in provider.generator.named] == ["qwen", "gemma"]
    monkeypatch.delenv("MODEL_POOL")
    single = build_tiered_provider("twitter", str(tmp_path), {"agent_configs": []}, score_fn=lambda t: 0.5)
    assert single.generator is None


def test_pooled_system_one_client(monkeypatch):
    from app.model_pool import EnsembleBackend
    from app.system_one.backends import LocalReadoutBackend
    from app.system_one.client import get_system_one_client

    monkeypatch.setenv("MODEL_POOL", json.dumps(POOL))
    monkeypatch.setenv("SYSTEM_ONE_ENSEMBLE", "s,stance")
    client = get_system_one_client()
    assert isinstance(client.backend, EnsembleBackend)
    assert client.backend.scope == {"s", "stance"}
    assert [b.prompt_format for b in client.backend.backends] == ["chatml", "plain"]
    monkeypatch.delenv("SYSTEM_ONE_ENSEMBLE")
    assert isinstance(get_system_one_client().backend, LocalReadoutBackend)


def test_bias_measures():
    from scripts.calibrate_model import engage_bias, polarity_bias

    records = [
        {"category": "post_sentiment", "label": "neutral", "keys": ["positive", "neutral", "negative"],
         "probs": [1.0, 0.0, 0.0]},
        {"category": "post_stance", "label": "other", "keys": ["support", "oppose", "neutral", "other"],
         "probs": [0.0, 0.0, 0.0, 1.0]},
        {"category": "will_engage", "label": True, "keys": [], "probs": [0.9, 0.1]},
        {"category": "will_engage", "label": False, "keys": [], "probs": [0.5, 0.5]},
    ]
    assert polarity_bias(records) == pytest.approx(1.0)
    assert engage_bias(records) == pytest.approx(0.2)


def test_calibrate_model_averages_an_ensemble(monkeypatch):
    from app.system_one.models import ChoiceAnswer
    from scripts import calibrate_model as cm

    class Fixed:
        def __init__(self, p_first):
            self.p_first = p_first

        def _ask_one(self, state, question):
            keys = list(question.criteria)
            probs = {k: (self.p_first if i == 0 else (1 - self.p_first) / (len(keys) - 1)) for i, k in enumerate(keys)}
            return ChoiceAnswer(choice=max(probs, key=probs.get), probabilities=probs, confidence=max(probs.values())), None, None

    assert cm.parse_with("http://127.0.0.1:8002/v1,phi-4-mini,plain") == ("http://127.0.0.1:8002/v1", "phi-4-mini", "plain")
    assert cm.parse_with("http://h/v1,m")[2] == "chatml"
    with pytest.raises(ValueError):
        cm.parse_with("only-a-url")
    row = {"type": "choice", "instructions": "?", "criteria": {"a": "A", "b": "B"}, "label": "a"}
    answer = cm.ask_all([Fixed(0.9), Fixed(0.3)], "state", row)
    assert answer.probabilities["a"] == pytest.approx(0.6)
