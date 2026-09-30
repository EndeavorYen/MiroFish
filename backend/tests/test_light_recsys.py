"""The light recommender keeps the simulation off torch/transformers (#65)."""

import json
import subprocess
import sys
from pathlib import Path

from app.simulation_policy import light_recsys

BACKEND = Path(__file__).resolve().parents[1]
HEAVY = ("torch", "transformers", "sentence_transformers", "sklearn")


def _user(uid, bio):
    return {"user_id": uid, "agent_id": uid, "bio": bio, "num_followers": 0}


def _post(pid, uid, content, created_at):
    return {"post_id": pid, "user_id": uid, "content": content, "created_at": created_at,
            "num_likes": 0, "num_dislikes": 0, "num_shares": 0}


def _run(users, posts, now, limit=2):
    light_recsys.reset_globals()
    return light_recsys.rec_sys_personalized_twh(
        users, posts, len(posts), [], [[] for _ in users], limit, now)


def test_few_posts_go_to_everyone():
    users = [_user(0, "a"), _user(1, "b")]
    posts = [_post(1, 0, "x", 0), _post(2, 1, "y", 0)]
    assert _run(users, posts, 1) == [[1, 2], [1, 2]]


def test_similar_posts_rank_first_and_own_posts_are_skipped():
    users = [_user(0, "電動車 電池 充電"), _user(1, "咖啡 烘焙 手沖"), _user(2, "旅遊")]
    posts = [
        _post(1, 2, "新的電池讓電動車充電更快", 1),
        _post(2, 2, "手沖咖啡的烘焙度怎麼選", 1),
        _post(3, 2, "今天天氣很好", 1),
        _post(4, 0, "我的電動車電池充電紀錄", 1),
    ]
    rec = _run(users, posts, 2)
    assert len(rec) == 3 and all(len(row) == 2 for row in rec)
    assert rec[0][0] == 1 and 4 not in rec[0]  # the matching post, never their own
    assert rec[1][0] == 2
    assert rec == _run(users, posts, 2)  # deterministic


def test_newer_posts_win_a_tie():
    users = [_user(0, "市場"), _user(1, "x")]
    posts = [_post(1, 1, "市場反應", 0), _post(2, 1, "市場反應", 50), _post(3, 1, "無關", 50)]
    assert _run(users, posts, 60)[0][0] == 2


def test_reddit_and_random_keep_the_oasis_shape():
    posts = [_post(i, 0, "p", "2026-10-01 00:00:00") for i in range(1, 4)]
    assert light_recsys.rec_sys_reddit(posts, [[], []], 5) == [[1, 2, 3], [1, 2, 3]]
    assert all(len(row) == 2 for row in light_recsys.rec_sys_random(posts, [[], []], 2))


def _loaded(code):
    out = subprocess.run([sys.executable, "-c", code], cwd=BACKEND, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=240)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_oasis_with_the_light_recommender_does_not_load_torch():
    loaded = _loaded(
        "import sys, json\n"
        "from app.simulation_policy import light_recsys\n"
        "assert light_recsys.install({'SIM_RECSYS': 'light'})\n"
        "import oasis\n"
        f"print(json.dumps([m for m in {HEAVY!r} if m in sys.modules]))\n"
    )
    assert loaded == []


def test_the_backend_does_not_load_the_simulation_stack():
    loaded = _loaded(
        "import sys, json\n"
        "from app import create_app\n"
        "create_app()\n"
        f"print(json.dumps([m for m in {HEAVY + ('oasis', 'camel')!r} if m in sys.modules]))\n"
    )
    assert loaded == []


def test_install_follows_the_switch():
    assert light_recsys.install({}) is False  # no profile: OASIS as shipped
    assert light_recsys.install({"SIM_RECSYS": "oasis"}) is False


def test_local_profiles_choose_the_light_recommender():
    from app.profiles import PROFILES

    assert {name: p.get("SIM_RECSYS") for name, p in PROFILES.items()} == {
        "local": "light", "local-hybrid": "light", "local-llm": "light"}


def test_neo4j_loads_only_when_a_graph_database_is_opened():
    loaded = _loaded(
        "import sys, json\n"
        "from app.simulation_policy.lazy_imports import defer_neo4j\n"
        "defer_neo4j()\n"
        "import oasis\n"
        "before = 'neo4j._async' in sys.modules\n"
        "from neo4j import GraphDatabase\n"
        "try:\n"
        "    GraphDatabase.driver('bolt://127.0.0.1:1', auth=('a', 'b'))\n"
        "except Exception:\n"
        "    pass\n"
        "print(json.dumps([before, 'neo4j._async' in sys.modules]))\n"
    )
    assert loaded == [False, True]
