import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.system_one.backends import HttpBackend
from app.system_one.client import SystemOneClient, get_system_one_client
from app.system_one.models import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
)


class _Handler(BaseHTTPRequestHandler):
    received = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).received.append(
            {"path": self.path, "auth": self.headers.get("Authorization"), "body": body}
        )
        payload = {
            "model": "jev-1.13.0",
            "answers": {
                "department": {
                    "type": "choice",
                    "choice": "technical",
                    "probabilities": {"billing": 0.15, "technical": 0.85},
                    "confidence": 0.82,
                },
                "anger": {
                    "type": "score",
                    "score": 1.035,
                    "probabilities": {"calm": 0.1, "frustrated": 0.765, "angry": 0.135},
                    "confidence": 0.7,
                },
                "urgent": {"type": "noul", "noul": 0.91},
            },
            "usage": {"input_tokens": 312, "output_tokens": 48},
        }
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def mock_server():
    _Handler.received = []
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _request():
    return SystemOneRequest(
        state="Help! My payouts have been failing for 3 days.",
        model="jev-latest",
        questions={
            "department": ChoiceQuestion(
                instructions="Which team should handle this?",
                criteria={"billing": "Payments", "technical": "Bugs"},
            ),
            "anger": ScoreQuestion(
                instructions="How frustrated", criteria=["calm", "frustrated", "angry"]
            ),
            "urgent": NoulQuestion(instructions="Is it urgent?"),
        },
    )


def test_http_backend_contract(mock_server):
    client = SystemOneClient(HttpBackend(base_url=mock_server, api_key="sk-test"))
    response = client.ask(_request())

    sent = _Handler.received[0]
    assert sent["path"] == "/v1/systemone"
    assert sent["auth"] == "Bearer sk-test"
    assert sent["body"] == {
        "state": "Help! My payouts have been failing for 3 days.",
        "model": "jev-latest",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "Payments", "technical": "Bugs"},
            },
            "anger": {
                "type": "score",
                "instructions": "How frustrated",
                "criteria": ["calm", "frustrated", "angry"],
            },
            "urgent": {"type": "noul", "instructions": "Is it urgent?"},
        },
    }

    assert response.model == "jev-1.13.0"
    department = response.answers["department"]
    assert isinstance(department, ChoiceAnswer)
    assert department.choice == "technical"
    assert department.confidence == 0.82
    anger = response.answers["anger"]
    assert isinstance(anger, ScoreAnswer)
    assert anger.score == 1.035
    urgent = response.answers["urgent"]
    assert isinstance(urgent, NoulAnswer)
    assert urgent.noul == 0.91
    assert response.usage.input_tokens == 312


def test_http_backend_accepts_base_url_with_path(mock_server):
    HttpBackend(base_url=mock_server + "/v1/systemone").ask(_request())
    assert _Handler.received[0]["path"] == "/v1/systemone"


def test_factory_selects_backend(monkeypatch):
    from app.config import Config
    from app.system_one.backends import LocalReadoutBackend

    monkeypatch.setattr(Config, "SYSTEM_ONE_BACKEND", "http")
    monkeypatch.setattr(Config, "SYSTEM_ONE_BASE_URL", "http://jev.example")
    assert isinstance(get_system_one_client().backend, HttpBackend)

    monkeypatch.setattr(Config, "SYSTEM_ONE_BACKEND", "local")
    monkeypatch.setattr(Config, "SYSTEM_ONE_BASE_URL", "http://127.0.0.1:8000/v1")
    assert isinstance(get_system_one_client().backend, LocalReadoutBackend)

    monkeypatch.setattr(Config, "SYSTEM_ONE_BACKEND", "bogus")
    with pytest.raises(ValueError):
        get_system_one_client()
