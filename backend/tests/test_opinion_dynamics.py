"""Opinion dynamics (#59): a role's stance moves with what it reads."""

import json
import threading

import pytest

from app.simulation_policy.opinion import OpinionParams, OpinionState, bounded_update, feed_texts, load_params

STANCES = {"太貴了，反對": 0.0, "還可以接受": 0.5, "很支持這個方案": 1.0, "有點擔心票價": 0.25, "期待搭乘": 0.75}


def _feed(*texts, author=2):
    return [{"post_id": i, "user_id": author, "content": t, "comments": []} for i, t in enumerate(texts)]


def test_without_interaction_the_stance_stays():
    assert bounded_update(0.4, [], mu=0.3, radius=0.3, stubbornness=0.5) == (0.4, 0)
    # Posts further than the radius do not pull (bounded confidence).
    assert bounded_update(0.0, [0.75, 1.0], mu=0.3, radius=0.3, stubbornness=0.5) == (0.0, 0)
    # A fully stubborn role does not move.
    assert bounded_update(0.4, [0.5], mu=0.3, radius=0.3, stubbornness=1.0) == (0.4, 1)


def test_with_interaction_it_moves_towards_close_opinions():
    after, close = bounded_update(0.5, [0.25, 0.75, 1.0], mu=0.5, radius=0.3, stubbornness=0.0)
    assert close == 2  # 1.0 is too far
    assert after == pytest.approx(0.5)  # 0.25 and 0.75 cancel
    after, _ = bounded_update(0.5, [0.75], mu=0.5, radius=0.3, stubbornness=0.5)
    assert after == pytest.approx(0.5 + 0.5 * 0.5 * 0.25)
    assert 0.5 < after < 0.75  # towards, never past


def test_a_role_reads_its_feed_without_its_own_posts():
    feed = [{"post_id": 1, "user_id": 1, "content": "我自己的貼文", "comments": [{"user_id": 2, "content": "別人的留言"}]},
            {"post_id": 2, "user_id": 3, "content": "別人的貼文", "comments": [{"user_id": 1, "content": "我的留言"}]}]
    assert feed_texts(feed, agent_id=1) == ["別人的留言", "別人的貼文"]


def test_the_state_updates_once_a_round_and_scores_each_text_once():
    scored = []

    def score(text):
        scored.append(text)
        return STANCES[text]

    state = OpinionState({1: 0.5, 2: 0.5}, OpinionParams(mu=0.5, radius=0.3, stubbornness=0.0), score)
    change = state.update(1, round_num=3, feed=_feed("期待搭乘", "還可以接受", "太貴了，反對", author=9))
    assert change == {"before": 0.5, "after": pytest.approx(0.5625), "read": 3, "close": 2}
    assert state.update(1, round_num=3, feed=_feed("很支持這個方案", author=9)) is None  # once a round
    state.update(2, round_num=3, feed=_feed("期待搭乘", author=9))
    assert scored.count("期待搭乘") == 1  # one readout per distinct text, shared
    assert state.update(99, round_num=3, feed=_feed("期待搭乘", author=9)) is None  # no stance to move
    assert state.stance(1) == pytest.approx(0.5625)


def test_stubbornness_follows_the_entity_type():
    params = OpinionParams(mu=1.0, radius=1.0, stubbornness=0.0, by_type={"GovernmentAgency": 1.0})
    state = OpinionState({1: 0.0, 2: 0.0}, params, lambda text: 1.0, entity_types={1: "GovernmentAgency", 2: "Person"})
    state.update(1, 0, _feed("a", author=9))
    state.update(2, 0, _feed("a", author=9))
    assert (state.stance(1), state.stance(2)) == (0.0, 1.0)


def test_updates_do_not_depend_on_the_order_of_concurrent_decisions():
    """A seeded run replays: each role reads posts, not other roles' states."""

    feeds = {agent: _feed(*list(STANCES)[agent % 5:] + list(STANCES)[:agent % 5], author=99) for agent in range(12)}

    def run(order, parallel):
        state = OpinionState({a: (a % 5) / 4 for a in range(12)}, OpinionParams(mu=0.4, radius=0.3, stubbornness=0.2),
                             lambda text: STANCES[text])
        for round_num in range(3):
            if parallel:
                threads = [threading.Thread(target=state.update, args=(a, round_num, feeds[a])) for a in order]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
            else:
                for a in order:
                    state.update(a, round_num, feeds[a])
        return [state.stance(a) for a in range(12)]

    forward = run(list(range(12)), parallel=False)
    assert run(list(reversed(range(12))), parallel=False) == forward
    assert run(list(range(12)), parallel=True) == forward


def test_parameters_are_off_by_default_and_read_from_env_and_file(tmp_path):
    assert load_params({}) is None
    assert load_params({"SIM_OPINION_DYNAMICS": "off"}) is None
    params_file = tmp_path / "opinion_params.json"
    params_file.write_text(json.dumps({"mu": 0.2, "stubbornness_by_type": {"MediaOutlet": 0.8}}), encoding="utf-8")
    params = load_params({"SIM_OPINION_DYNAMICS": "bounded", "SIM_OPINION_RADIUS": "0.4"}, path=str(params_file))
    assert (params.mu, params.radius, params.stubbornness) == (0.2, 0.4, 0.5)  # file, env, default
    assert params.stubbornness_of("MediaOutlet") == 0.8 and params.stubbornness_of("Person") == 0.5
    with pytest.raises(ValueError):
        load_params({"SIM_OPINION_DYNAMICS": "degroot"})
    with pytest.raises(ValueError):
        load_params({"SIM_OPINION_DYNAMICS": "bounded", "SIM_OPINION_MU": "2"}, path=str(tmp_path / "none.json"))


def test_the_policy_uses_the_moving_stance(tmp_path):
    from tests.test_simulation_policy import FakeSystemOne, _obs
    from app.simulation_policy.policy import DecisionLog, SystemOnePolicy
    from app.simulation_policy.taxonomy import load_taxonomy

    state = OpinionState({1: 0.0}, OpinionParams(mu=1.0, radius=1.0, stubbornness=0.0), lambda text: 1.0)
    log_path = tmp_path / "decisions.jsonl"
    policy = SystemOnePolicy(FakeSystemOne(), load_taxonomy("twitter"), seed=7, stance_prior={1: 0.0},
                             decision_log=DecisionLog(str(log_path)), opinion=state)
    policy.decide(_obs(agent_id=1, round_num=0))
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["opinion"]["before"] == 0.0 and record["opinion"]["after"] == 1.0
    assert policy._stance(1) == 1.0  # the intent and the action tree now use it

    still = SystemOnePolicy(FakeSystemOne(), load_taxonomy("twitter"), seed=7, stance_prior={1: 0.0})
    still.decide(_obs(agent_id=1, round_num=0))
    assert still._stance(1) == 0.0  # off: the prep's stance all run


def test_build_policy_turns_it_on_with_the_env(tmp_path, monkeypatch):
    from app.simulation_policy import oasis_bridge

    config = {"simulation_requirement": "模擬北港市調漲水費後的反應。",
              "agent_configs": [{"agent_id": 1, "sentiment_bias": 0.5, "entity_type": "Person"}]}
    assert oasis_bridge.opinion_state(config, {1: 0.75}, score=lambda t: 0.5) is None
    monkeypatch.setenv("SIM_OPINION_DYNAMICS", "bounded")
    state = oasis_bridge.opinion_state(config, {1: 0.75}, score=lambda t: 0.5)
    assert state.stance(1) == 0.75
