"""GET /api/report/<id>/metrics: what the result page shows (#66)."""

import json
from pathlib import Path

from app.services.report_manager import ReportManager


def _client():
    from app import create_app

    return create_app().test_client()


def test_the_metrics_of_a_report_are_served():
    folder = Path(ReportManager.REPORTS_DIR) / "report_0123abcd"  # conftest points it into tmp
    folder.mkdir(parents=True)
    metrics = {"scan": {"main_camp": {"value": "support", "confidence": "high"}}, "spread": {"twitter": []}}
    (folder / "report_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")

    response = _client().get("/api/report/report_0123abcd/metrics")
    assert response.status_code == 200
    assert response.get_json()["data"] == metrics


def test_a_report_without_metrics_or_a_bad_id_is_404():
    client = _client()
    assert client.get("/api/report/report_ffffffff/metrics").status_code == 404
    for bad in ("..%2F..%2Fsecret", "report_XYZ", "notareport"):
        assert client.get(f"/api/report/{bad}/metrics").status_code == 404
