"""System One decision policy (#9) with a fake System One client."""

import asyncio
import hashlib
import json
import math
from dataclasses import dataclass

import pytest

from app.simulation_policy.content import ContentIntent, TemplateContentProvider
from app.simulation_policy.emotion import (
    AgentStateStore,
    neutral_state,
    score_to_unit,
    update_emotion,
)
from app.simulation_policy.policy import DecisionLog, Observation, SystemOnePolicy
from app.simulation_policy.taxonomy import TaxonomyError, load_taxonomy, parse_taxonomy
from app.system_one.models import (
    ChoiceAnswer,
    ChoiceQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneResponse,
)

TWITTER_ARGS = {
    "CREATE_POST": {"content"},
    "LIKE_POST": {"post_id"},
    "REPOST": {"post_id"},
    "QUOTE_POST": {"post_id", "quote_content"},
    "FOLLOW": {"followee_id"},
    "DO_NOTHING": set(),
}
REDDIT_ARGS = {
    "LIKE_POST": {"post_id"},
    "DISLIKE_POST": {"post_id"},
    "CREATE_POST": {"content"},
    "CREATE_COMMENT": {"post_id", "content"},
    "LIKE_COMMENT": {"comment_id"},
    "DISLIKE_COMMENT": {"comment_id"},
    "SEARCH_POSTS": {"query"},
    "SEARCH_USER": {"query"},
    "TREND": set(),
    "REFRESH": set(),
    "DO_NOTHING": set(),
    "FOLLOW": {"followee_id"},
    "MUTE": {"mutee_id"},
}


class FakeSystemOne:
    """Deterministic, state-dependent distributions; no model."""

    def __init__(self, force: dict[str, str] | None = None):
        self.calls = 0
        self.force = force or {}

    def ask(self, request):
        answers = {}
        for name, question in request.questions.items():
            self.calls += 1
            digest = hashlib.sha256((request.state + name).encode()).digest()
            if isinstance(question, ChoiceQuestion):
                keys = list(question.criteria)
                forced = next((self.force[k] for k in keys if k in self.force), None)
                if forced is None and name in self.force:
                    forced = self.force[name]
                weights = [1 + digest[i % len(digest)] for i in range(len(keys))]
                if forced in keys:
                    weights = [1000 if k == forced else 1 for k in keys]
                total = sum(weights)
                probs = {k: w / total for k, w in zip(keys, weights)}
                best = max(probs, key=probs.get)
                answers[name] = ChoiceAnswer(choice=best, probabilities=probs, confidence=probs[best])
            elif isinstance(question, ScoreQuestion):
                n = len(question.criteria)
                score = (digest[0] / 255) * (n - 1)
                answers[name] = ScoreAnswer(
                    score=score,
                    probabilities={c: 1 / n for c in question.criteria},
                    confidence=1 / n,
                )
        return SystemOneResponse(model="fake", answers=answers)


def _feed():
    return [
        {
            "post_id": 101,
            "user_id": 2,
            "content": "凌雲飛行的票價太貴了",
            "comments": [{"comment_id": 501, "user_id": 3, "content": "同意"}],
        },
        {"post_id": 102, "user_id": 3, "content": "試飛很順利，期待搭乘", "comments": []},
    ]


def _obs(platform="twitter", agent_id=1, round_num=0, feed=None):
    return Observation(
        platform=platform,
        round_num=round_num,
        agent_id=agent_id,
        agent_name="Alice",
        persona="東海市通勤族，關心票價",
        feed=_feed() if feed is None else feed,
        user_names={1: "Alice", 2: "Bob", 3: "Carol"},
        topics=["空中計程車票價"],
    )


def _policy(platform="twitter", seed=7, force=None, tmp_path=None, alpha=0.7):
    log = DecisionLog(str(tmp_path / "decisions.jsonl")) if tmp_path else None
    store = AgentStateStore(str(tmp_path / "agent_state.db")) if tmp_path else None
    return SystemOnePolicy(
        FakeSystemOne(force),
        load_taxonomy(platform),
        seed=seed,
        alpha=alpha,
        state_store=store,
        decision_log=log,
    )


def _leaf_forces(platform):
    """force map that drives the tree to each leaf."""

    taxonomy = load_taxonomy(platform)
    forces = []
    for path, leaf in taxonomy.leaves.items():
        force = {}
        # Force each level's key; option keys are unique enough per level.
        for key in path:
            force[key] = key
        forces.append((path, leaf, force))
    return forces


@pytest.mark.parametrize("platform,expected", [("twitter", TWITTER_ARGS), ("reddit", REDDIT_ARGS)])
def test_every_action_produces_valid_manual_action_args(platform, expected):
    taxonomy = load_taxonomy(platform)
    assert taxonomy.actions == set(expected)
    feed = _feed()
    post_ids = {p["post_id"] for p in feed}
    comment_ids = {c["comment_id"] for p in feed for c in p["comments"]}
    user_ids = {p["user_id"] for p in feed}
    seen = set()
    for path, leaf, force in _leaf_forces(platform):
        class PathForcing(FakeSystemOne):
            def ask(self, request, _path=path):
                response = super().ask(request)
                for name, answer in response.answers.items():
                    if isinstance(answer, ChoiceAnswer):
                        keys = list(answer.probabilities)
                        target = next((k for k in _path if k in keys), None)
                        if target:
                            probs = {k: (0.999 if k == target else 0.001 / (len(keys) - 1)) for k in keys}
                            response.answers[name] = ChoiceAnswer(
                                choice=target, probabilities=probs, confidence=0.999
                            )
                return response

        policy = SystemOnePolicy(PathForcing(), taxonomy, seed=3)
        decision = policy.decide(_obs(platform))
        assert decision.action == leaf.action, path
        assert set(decision.args) == expected[decision.action], (path, decision.args)
        if "post_id" in decision.args:
            assert decision.args["post_id"] in post_ids
        if "comment_id" in decision.args:
            assert decision.args["comment_id"] in comment_ids
        for key in ("followee_id", "mutee_id"):
            if key in decision.args:
                assert decision.args[key] in user_ids and decision.args[key] != 1
        for key in ("content", "quote_content", "query"):
            if key in decision.args:
                assert isinstance(decision.args[key], str) and decision.args[key]
        seen.add(decision.action)
    assert seen == set(expected)


def test_empty_feed_falls_back_to_do_nothing_for_targeted_actions():
    taxonomy = load_taxonomy("twitter")

    class ForceLike(FakeSystemOne):
        def ask(self, request):
            response = super().ask(request)
            for name, answer in response.answers.items():
                if isinstance(answer, ChoiceAnswer) and "engage" in answer.probabilities:
                    keys = list(answer.probabilities)
                    response.answers[name] = ChoiceAnswer(
                        choice="engage",
                        probabilities={k: 1.0 if k == "engage" else 0.0 for k in keys},
                        confidence=1.0,
                    )
                if isinstance(answer, ChoiceAnswer) and "like_post" in answer.probabilities:
                    keys = list(answer.probabilities)
                    response.answers[name] = ChoiceAnswer(
                        choice="like_post",
                        probabilities={k: 1.0 if k == "like_post" else 0.0 for k in keys},
                        confidence=1.0,
                    )
            return response

    decision = SystemOnePolicy(ForceLike(), taxonomy, seed=1).decide(_obs(feed=[]))
    assert decision.action == "DO_NOTHING" and decision.args == {}
    assert decision.record["fallback"] == "no post to act on"


def test_emotion_update_math():
    prev = {"anger": 0.2, "joy": 0.8}
    readout = {"anger": 1.0, "joy": 0.0}
    assert update_emotion(prev, readout, 1.0) == prev
    assert update_emotion(prev, readout, 0.0) == readout
    mixed = update_emotion(prev, readout, 0.75)
    assert math.isclose(mixed["anger"], 0.75 * 0.2 + 0.25 * 1.0)
    assert math.isclose(mixed["joy"], 0.75 * 0.8)
    with pytest.raises(ValueError):
        update_emotion(prev, readout, 1.5)
    assert score_to_unit(0) == 0.0 and score_to_unit(4) == 1.0 and score_to_unit(2) == 0.5


def test_emotion_persists_with_inertia_across_rounds(tmp_path):
    policy = _policy(tmp_path=tmp_path, alpha=0.5)
    first = policy.decide(_obs(round_num=0)).record
    second = policy.decide(_obs(round_num=1)).record
    for dim, value in second["emotion"].items():
        expected = 0.5 * first["emotion"][dim] + 0.5 * second["emotion_readout"][dim]
        assert math.isclose(value, expected, abs_tol=1e-5)
    history = AgentStateStore(str(tmp_path / "agent_state.db")).history("twitter", 1)
    assert [r for r, _ in history] == [0, 1]
    assert first["emotion"] != neutral_state()


def test_decisions_replay_exactly_with_the_same_seed(tmp_path):
    def run(directory, seed):
        directory.mkdir()
        policy = _policy(platform="reddit", seed=seed, tmp_path=directory)
        for round_num in range(4):
            for agent_id in (1, 2, 3):
                policy.decide(_obs(platform="reddit", agent_id=agent_id, round_num=round_num))
        return (directory / "decisions.jsonl").read_text(encoding="utf-8")

    a = run(tmp_path / "a", 11)
    b = run(tmp_path / "b", 11)
    c = run(tmp_path / "c", 12)
    assert a == b
    assert a != c
    rows = [json.loads(line) for line in a.splitlines()]
    assert len(rows) == 12
    for row in rows:
        assert {"round", "agent_id", "path", "options", "probs", "chosen", "state_hash"} <= set(row)


def test_sampling_does_not_collapse_to_argmax():
    policy = _policy(platform="twitter", seed=5)
    actions = {policy.decide(_obs(agent_id=i, round_num=r)).action for i in range(8) for r in range(3)}
    assert len(actions) >= 3


def test_taxonomy_validation():
    good = {"twitter": {"options": {"x": {"action": "DO_NOTHING"}, "none": {"action": "DO_NOTHING"}}}}
    assert parse_taxonomy(good, "twitter").leaves
    with pytest.raises(TaxonomyError):
        parse_taxonomy({"twitter": {"options": {"x": {"action": "DO_NOTHING"}}}}, "twitter")
    too_many = {f"o{i}": {"action": "DO_NOTHING"} for i in range(26)}
    too_many["none"] = {"action": "DO_NOTHING"}
    with pytest.raises(TaxonomyError):
        parse_taxonomy({"twitter": {"options": too_many}}, "twitter")
    with pytest.raises(TaxonomyError):
        parse_taxonomy(
            {"twitter": {"options": {"x": {"action": "LIKE_POST", "needs": ["telepathy"]}, "none": {"action": "DO_NOTHING"}}}},
            "twitter",
        )
    for platform in ("twitter", "reddit"):
        taxonomy = load_taxonomy(platform)
        assert "other" in taxonomy.dialogue_kinds


def test_template_content_provider_uses_intent():
    text = TemplateContentProvider().generate(
        ContentIntent(kind="criticism", stance=0.0, intensity=0.9, target_ref="1", persona_ref=1, topic="票價")
    )
    assert "票價" in text and "非常反對" in text and text.endswith("！")


def test_oasis_bridge_builds_manual_actions(tmp_path):
    from oasis import ActionType, ManualAction

    from app.simulation_policy.oasis_bridge import system_one_actions

    @dataclass
    class Info:
        name: str
        user_name: str = ""
        description: str = "市民"
        profile: dict = None

    class FakeActionApi:
        async def refresh(self):
            return {"success": True, "posts": _feed()}

    class FakeAgentEnv:
        action = FakeActionApi()

    class FakeAgent:
        def __init__(self, agent_id, name):
            self.social_agent_id = agent_id
            self.user_info = Info(name=name)
            self.env = FakeAgentEnv()

    agents = {1: FakeAgent(1, "Alice"), 2: FakeAgent(2, "Bob"), 3: FakeAgent(3, "Carol")}

    class FakeGraph:
        def get_agents(self):
            return list(agents.items())

    class FakeEnv:
        agent_graph = FakeGraph()

    policy = _policy(platform="twitter", seed=9, tmp_path=tmp_path)
    actions = asyncio.run(
        system_one_actions(FakeEnv(), [(1, agents[1]), (2, agents[2])], policy, "twitter", 0)
    )
    assert set(actions) == {agents[1], agents[2]}
    for action in actions.values():
        assert isinstance(action, ManualAction)
        assert isinstance(action.action_type, ActionType)
    rows = (tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2


def test_oasis_bridge_degrades_a_failed_decision_to_do_nothing(tmp_path):
    from oasis import ActionType

    from app.simulation_policy.oasis_bridge import system_one_actions

    class Broken(FakeSystemOne):
        def ask(self, request):
            raise TimeoutError("timed out")

    class FakeActionApi:
        async def refresh(self):
            return {"success": True, "posts": _feed()}

    class FakeAgent:
        def __init__(self, agent_id):
            self.social_agent_id = agent_id
            self.user_info = None
            self.env = type("E", (), {"action": FakeActionApi()})()

    agent = FakeAgent(1)
    env = type("Env", (), {"agent_graph": type("G", (), {"get_agents": lambda self: [(1, agent)]})()})()
    policy = SystemOnePolicy(
        Broken(), load_taxonomy("twitter"), seed=1, decision_log=DecisionLog(str(tmp_path / "d.jsonl"))
    )
    actions = asyncio.run(system_one_actions(env, [(1, agent)], policy, "twitter", 0))
    assert actions[agent].action_type == ActionType.DO_NOTHING
    row = json.loads((tmp_path / "d.jsonl").read_text(encoding="utf-8"))
    assert row["action"] == "DO_NOTHING" and row["error"].startswith("TimeoutError")


def test_build_policy_shares_stores_and_validates_settings(tmp_path, monkeypatch):
    from app.simulation_policy.oasis_bridge import build_policy

    twitter = build_policy("twitter", str(tmp_path), seed=1, client=FakeSystemOne(), content_provider=TemplateContentProvider())
    reddit = build_policy("reddit", str(tmp_path), seed=2, client=FakeSystemOne(), content_provider=TemplateContentProvider())
    assert twitter.state_store is reddit.state_store
    assert twitter.log is reddit.log
    with pytest.raises(ValueError):
        build_policy("twitter", str(tmp_path), seed=1, client=FakeSystemOne(), alpha=1.5, content_provider=TemplateContentProvider())
    monkeypatch.setenv("SIM_DECISION_CONCURRENCY", "zero")
    with pytest.raises(ValueError):
        build_policy("twitter", str(tmp_path), seed=1, client=FakeSystemOne(), content_provider=TemplateContentProvider())


def test_bridge_refreshes_recommendations_before_observing(tmp_path):
    from app.simulation_policy.oasis_bridge import system_one_actions

    calls = []

    class FakePlatform:
        async def update_rec_table(self):
            calls.append("rec")

    class FakeActionApi:
        async def refresh(self):
            calls.append("refresh")
            return {"success": True, "posts": list(reversed(_feed()))}

    class FakeAgent:
        def __init__(self):
            self.social_agent_id = 1
            self.user_info = None
            self.env = type("E", (), {"action": FakeActionApi()})()

    agent = FakeAgent()
    env = type("Env", (), {
        "platform": FakePlatform(),
        "agent_graph": type("G", (), {"get_agents": lambda self: [(1, agent)]})(),
    })()
    policy = _policy(platform="twitter", seed=1, tmp_path=tmp_path)
    asyncio.run(system_one_actions(env, [(1, agent)], policy, "twitter", 1))
    assert calls[:2] == ["rec", "refresh"]


class Pick(FakeSystemOne):
    """Picks a fixed option wherever it is offered (e.g. engage, like_post)."""

    def __init__(self, *prefer):
        super().__init__()
        self.prefer = prefer

    def ask(self, request):
        response = super().ask(request)
        for name, answer in response.answers.items():
            if isinstance(answer, ChoiceAnswer):
                pick = next((p for p in self.prefer if p in answer.probabilities), None)
                if pick:
                    keys = list(answer.probabilities)
                    response.answers[name] = ChoiceAnswer(
                        choice=pick, probabilities={k: 1.0 if k == pick else 0.0 for k in keys}, confidence=1.0
                    )
        return response


def test_decide_round_takes_extra_actions_and_reuses_emotion(tmp_path):
    from app.simulation_policy.policy import MAX_ACTIONS_PER_ROUND

    policy = SystemOnePolicy(Pick("engage", "like_post"), load_taxonomy("twitter"), seed=3, extra_action_rate=0.99,
                             decision_log=DecisionLog(str(tmp_path / "d.jsonl")))
    feed = _feed() + [{"post_id": 103, "user_id": 2, "content": "票價何時公布？", "comments": []}]
    decisions = policy.decide_round(_obs(feed=feed))
    assert len(decisions) == MAX_ACTIONS_PER_ROUND
    assert all(d.action == "LIKE_POST" for d in decisions)
    rows = [json.loads(line) for line in (tmp_path / "d.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r.get("action_index", 0) for r in rows] == [0, 1, 2]
    assert len({json.dumps(r["emotion"], sort_keys=True) for r in rows}) == 1  # one emotion update

    single = SystemOnePolicy(Pick("engage", "like_post"), load_taxonomy("twitter"), seed=3)
    assert len(single.decide_round(_obs())) == 1  # extra_action_rate defaults to 0
    idle = SystemOnePolicy(Pick("none"), load_taxonomy("twitter"), seed=3, extra_action_rate=0.99)
    assert [d.action for d in idle.decide_round(_obs())] == ["DO_NOTHING"]
    with pytest.raises(ValueError):
        SystemOnePolicy(FakeSystemOne(), load_taxonomy("twitter"), extra_action_rate=1.0)


def test_later_actions_in_a_round_skip_used_targets():
    policy = SystemOnePolicy(Pick("engage", "like_post"), load_taxonomy("twitter"), seed=3, extra_action_rate=0.99)
    decisions = policy.decide_round(_obs())  # the feed has two posts
    liked = [d.args["post_id"] for d in decisions if d.action == "LIKE_POST"]
    assert sorted(liked) == [101, 102]  # the third like finds no unused post and stops the round
    assert len(decisions) == 2


def test_one_post_per_round_and_failures_keep_earlier_actions(tmp_path):
    feed = _feed() + [{"post_id": 103, "user_id": 2, "content": "票價何時公布？", "comments": []}]
    poster = SystemOnePolicy(Pick("create", "create_post"), load_taxonomy("twitter"), seed=3, extra_action_rate=0.99)
    assert [d.action for d in poster.decide_round(_obs(feed=feed))] == ["CREATE_POST"]

    class FailsAfterFirst(Pick):
        def __init__(self):
            super().__init__("engage", "like_post")
            self.decisions = 0

        def ask(self, request):
            if "這一輪已經做了" in request.state:
                raise TimeoutError("model server down")
            return super().ask(request)

    policy = SystemOnePolicy(FailsAfterFirst(), load_taxonomy("twitter"), seed=3, extra_action_rate=0.99)
    assert [d.action for d in policy.decide_round(_obs(feed=feed))] == ["LIKE_POST"]


def test_interview_ignores_exit_rows_of_extra_actions(tmp_path):
    from app.simulation_policy.interview import interview_context

    rows = [
        {"agent_id": 1, "platform": "twitter", "action": "LIKE_POST", "args": {}},
        {"agent_id": 1, "platform": "twitter", "action": "DO_NOTHING", "args": {}, "action_index": 1},
    ]
    (tmp_path / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    text = interview_context(str(tmp_path), 1)
    assert "旁观" not in text and "点赞" in text


def test_stance_prior_anchors_the_intent_stance():
    from app.simulation_policy.oasis_bridge import stance_priors

    assert stance_priors({"agent_configs": [
        {"agent_id": 1, "sentiment_bias": -0.8}, {"agent_id": 2, "sentiment_bias": 1.4}, {"agent_id": 3},
    ]}) == {1: pytest.approx(0.1), 2: 1.0}

    feed = _feed()
    plain = SystemOnePolicy(Pick("create", "create_post"), load_taxonomy("twitter"), seed=3)
    anchored = SystemOnePolicy(Pick("create", "create_post"), load_taxonomy("twitter"), seed=3,
                               stance_prior={1: 0.0}, stance_prior_weight=0.5)
    raw = plain.decide(_obs(feed=feed)).record["intent"]
    held = anchored.decide(_obs(feed=feed)).record["intent"]
    assert "stance_prior" not in raw
    assert held["stance_readout"] == pytest.approx(raw["stance"])
    assert held["stance"] == pytest.approx(0.5 * raw["stance"])  # halfway to the standing stance 0
    with pytest.raises(ValueError):
        SystemOnePolicy(FakeSystemOne(), load_taxonomy("twitter"), stance_prior_weight=1.5)
