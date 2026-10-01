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


def test_a_reading_that_fails_is_skipped_not_the_decision():
    def score(text):
        if text == "壞掉的貼文":
            raise TimeoutError("llama-server slot busy")
        return STANCES[text]

    state = OpinionState({1: 0.5}, OpinionParams(mu=0.5, radius=0.3, stubbornness=0.0), score)
    change = state.update(1, 0, _feed("壞掉的貼文", "期待搭乘", author=9))
    assert change["failed"] == 1 and change["read"] == 1 and change["after"] > 0.5
    only_failures = OpinionState({1: 0.5}, OpinionParams(), lambda text: 1 / 0)
    assert only_failures.update(1, 0, _feed("期待搭乘", author=9))["after"] == 0.5  # keeps its stance


def test_the_same_text_read_at_the_same_time_is_scored_once():
    import time

    calls = []

    def slow(text):
        calls.append(text)
        time.sleep(0.1)
        return 0.75

    state = OpinionState({a: 0.5 for a in range(6)}, OpinionParams(), slow)
    threads = [threading.Thread(target=state.update, args=(a, 0, _feed("期待搭乘", author=99))) for a in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls == ["期待搭乘"]
    assert len({state.stance(a) for a in range(6)}) == 1


def test_reposts_quotes_and_report_warnings_carry_the_words_read():
    feed = [
        {"post_id": 1, "user_id": 2, "content": "User 2 reposted a post from User 1. Repost content: 我說的話. ", "comments": []},
        {"post_id": 2, "user_id": 3, "content": "User 3 reposted a post from User 4. Repost content: 別人說的話. ", "comments": []},
        {"post_id": 3, "user_id": 5, "content": "User 5 quoted a post from User 1. Quote content: 我不同意. Original Content: 我說的話", "comments": []},
        {"post_id": 4, "user_id": 6, "content": "[Warning: This post has been reported 3 times]\n被檢舉的貼文", "comments": []},
    ]
    assert feed_texts(feed, agent_id=1) == ["別人說的話", "我不同意", "被檢舉的貼文"]


def test_one_stance_per_role_across_both_platforms():
    state = OpinionState({1: 0.0}, OpinionParams(mu=1.0, radius=1.0, stubbornness=0.0), lambda text: 1.0)
    assert state.update(1, 0, _feed("推特貼文", author=9), platform="twitter")["after"] == 1.0
    assert state.update(1, 0, _feed("推特貼文", author=9), platform="twitter") is None  # once a round per platform
    assert state.update(1, 0, _feed("論壇貼文", author=9), platform="reddit")["before"] == 1.0  # the same stance


def test_build_policy_shares_one_state_and_scores_with_the_policy_client(tmp_path, monkeypatch):
    from tests.test_simulation_policy import FakeSystemOne
    from app.simulation_policy import oasis_bridge

    monkeypatch.setenv("SIM_OPINION_DYNAMICS", "bounded")
    (tmp_path / "simulation_config.json").write_text(json.dumps({
        "simulation_requirement": "模擬北港市調漲水費後的反應。",
        "agent_configs": [{"agent_id": 1, "sentiment_bias": 0.0, "entity_type": "Person"}],
    }, ensure_ascii=False), encoding="utf-8")
    client = FakeSystemOne()
    twitter = oasis_bridge.build_policy("twitter", str(tmp_path), seed=1, client=client, content_provider=object())
    reddit = oasis_bridge.build_policy("reddit", str(tmp_path), seed=2, client=client, content_provider=object())
    assert twitter.opinion is reddit.opinion is not None
    twitter.opinion.update(1, 0, _feed("期待搭乘", author=9), platform="twitter")
    assert client.calls == 1  # the injected client scored it; no model service


def test_a_malformed_params_file_says_so(tmp_path):
    bad = tmp_path / "opinion_params.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        load_params({"SIM_OPINION_DYNAMICS": "true"}, path=str(bad))


def test_the_log_keeps_the_prep_stance_and_the_stance_used(tmp_path):
    from tests.test_simulation_policy import FakeSystemOne, _obs
    from app.simulation_policy.policy import DecisionLog, SystemOnePolicy
    from app.simulation_policy.taxonomy import load_taxonomy

    state = OpinionState({1: 0.0}, OpinionParams(mu=1.0, radius=1.0, stubbornness=0.0), lambda text: 1.0)
    log_path = tmp_path / "decisions.jsonl"
    taxonomy = load_taxonomy("twitter")
    path = next(p for p, leaf in taxonomy.leaves.items() if leaf.action == "CREATE_POST")
    policy = SystemOnePolicy(FakeSystemOne({key: key for key in path}), taxonomy,
                             seed=7, stance_prior={1: 0.0}, decision_log=DecisionLog(str(log_path)), opinion=state)
    for round_num in range(3):
        policy.decide(_obs(agent_id=1, round_num=round_num))
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert all(row["stance_state"] == 1.0 for row in rows)
    intents = [row["intent"] for row in rows if row.get("intent")]
    assert intents  # at least one post was written, so the intent fields are checked
    assert all(i["stance_prior"] == 0.0 and i["stance_state"] == 1.0 for i in intents)


def test_the_report_says_when_opinion_dynamics_was_on(tmp_path):
    from app.services.metrics_report import scan_conclusions

    config = {"simulation_requirement": "模擬北港市調漲水費後的反應。",
              "agent_configs": [{"agent_id": 1, "entity_name": "甲", "entity_type": "Person", "sentiment_bias": 0.0, "stance_raw": 0.5}]}
    (tmp_path / "simulation_config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "twitter").mkdir()
    post = {"round": 1, "agent_id": 1, "agent_name": "甲", "action_type": "CREATE_POST", "action_args": {"content": "太貴了"}}
    (tmp_path / "twitter" / "actions.jsonl").write_text(json.dumps(post, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "decisions.jsonl").write_text(json.dumps({"round": 1, "agent_id": 1, "stance_state": 0.4}), encoding="utf-8")
    scan = scan_conclusions(str(tmp_path), score_fn=lambda text, question: 0.5)
    assert scan["opinion_dynamics"] is True and "意見動態" in scan["main_camp"]["evidence"]
    (tmp_path / "decisions.jsonl").write_text(json.dumps({"round": 1, "agent_id": 1}), encoding="utf-8")
    assert scan_conclusions(str(tmp_path), score_fn=lambda text, question: 0.5)["opinion_dynamics"] is False
