"""Multi-seed local scan (#53)."""

from scripts import scan as sc


def _scan(camp, tendency, trend, by_role, indistinct=False):
    return {
        "main_camp": {"value": camp},
        "tendency": {"value": tendency},
        "trend": {"value": trend},
        "ranking": {"indistinct": indistinct},
        "by_role": by_role,
    }


def test_consistent_seeds_need_no_confirmation():
    roles = {"甲": 0.9, "乙": 0.6, "丙": 0.2, "丁": 0.4}
    row = sc.aggregate("s", [_scan("support", 0.7, 0.05, roles), _scan("support", 0.72, 0.02, roles)])
    assert row["main_camp"] == {"value": "support", "agree": 2}
    assert row["ranking"]["most_supportive"][0] == "甲" and row["ranking"]["most_opposed"][0] == "丙"
    assert row["ranking"]["stability"] == 1.0
    assert row["confirm"] == []
    assert "主要陣營、整體傾向可用" in sc.render([row])


def test_disagreeing_seeds_are_sent_to_confirmation():
    up = {"甲": 0.9, "乙": 0.6, "丙": 0.2, "丁": 0.4}
    down = {"甲": 0.2, "乙": 0.4, "丙": 0.9, "丁": 0.6}
    row = sc.aggregate("s", [
        _scan("support", 0.7, 0.1, up), _scan("oppose", 0.4, -0.2, down), _scan("support", 0.6, 0.1, up),
    ])
    reasons = "；".join(row["confirm"])
    assert "主要陣營只有 2/3" in reasons and "走向" in reasons and "不穩" in reasons
    assert "local-llm" in sc.render([row])


def test_indistinct_roles_are_reported_as_such():
    roles = {"甲": 0.5, "乙": 0.5, "丙": 0.5}
    row = sc.aggregate("s", [_scan("neutral", 0.5, 0.0, roles, True), _scan("neutral", 0.5, 0.0, roles, True)])
    assert row["confirm"] == ["角色立場無法區分"]
    assert "無法區分" in sc.render([row])
