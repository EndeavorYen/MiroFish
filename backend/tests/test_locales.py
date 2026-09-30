import json
from pathlib import Path
import pytest

from app.utils.locale import set_locale, get_locale, t
from app.services.metrics_report import render_markdown


LOCALES_DIR = Path(__file__).resolve().parents[2] / "locales"


def test_languages_registry_contains_zh_tw():
    registry_path = LOCALES_DIR / "languages.json"
    assert registry_path.exists()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert "zh-TW" in registry
    assert registry["zh-TW"]["label"] == "繁體中文"
    assert registry["zh-TW"]["llmInstruction"] == "請使用繁體中文回答。"


def _extract_keys(data, prefix=""):
    keys = set()
    for k, v in data.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            keys.update(_extract_keys(v, full_key))
        else:
            keys.add(full_key)
    return keys


def test_zh_tw_key_parity_with_zh():
    zh_path = LOCALES_DIR / "zh.json"
    zh_tw_path = LOCALES_DIR / "zh-TW.json"
    assert zh_tw_path.exists(), "locales/zh-TW.json must exist"

    zh_data = json.loads(zh_path.read_text(encoding="utf-8"))
    zh_tw_data = json.loads(zh_tw_path.read_text(encoding="utf-8"))

    zh_keys = _extract_keys(zh_data)
    zh_tw_keys = _extract_keys(zh_tw_data)

    missing_in_tw = zh_keys - zh_tw_keys
    extra_in_tw = zh_tw_keys - zh_keys

    assert not missing_in_tw, f"Keys in zh.json missing in zh-TW.json: {missing_in_tw}"
    assert not extra_in_tw, f"Keys in zh-TW.json missing in zh.json: {extra_in_tw}"


@pytest.fixture(autouse=True)
def reset_locale():
    yield
    set_locale("zh-TW")


def test_metrics_report_rendered_with_locale():
    metrics = {
        "agents": 2,
        "entity_types": {"市民": 2},
        "hot_topics": ["交通"],
        "actions": {"twitter": {"total": {"post": 10}}},
        "emotion": {},
        "spread": {"twitter": []},
        "stance": {"市民": {"expressed_stance_mean": 0.5, "configured_stances": "中立", "expressed_stance_samples": 2}},
    }

    try:
        set_locale("zh-TW")
        md_tw = render_markdown(metrics)
        assert "# 模擬指標報告" in md_tw
        assert "## 概況" in md_tw
        assert "## 動作分布" in md_tw

        set_locale("zh")
        md_zh = render_markdown(metrics)
        assert "# 模拟指标报告" in md_zh
        assert "## 概况" in md_zh
        assert "## 动作分布" in md_zh

        set_locale("en")
        md_en = render_markdown(metrics)
        assert "# Simulation Metrics Report" in md_en
        assert "## Overview" in md_en
        assert "## Action Distribution" in md_en
    finally:
        set_locale("zh-TW")

