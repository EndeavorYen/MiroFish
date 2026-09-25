import json
import math
import random

from app.system_one.backends import LocalReadoutBackend, parse_top_logprobs
from app.system_one.client import SystemOneClient
from app.system_one.models import ChoiceQuestion, SystemOneRequest
from app.system_one.tree import ask_tree


class FakeCompletions:
    """Return fixed label logprobs keyed by the question text in the prompt."""

    def __init__(self, table):
        self.table = table
        self.prompts = []

    def __call__(self, payload):
        prompt = payload["prompt"]
        self.prompts.append(prompt)
        for needle, top in self.table.items():
            if needle in prompt.rsplit("Question:", 1)[-1]:
                return _llama_response(top)
        return _llama_response({"A": math.log(0.5), "B": math.log(0.5)})


def _llama_response(top):
    return {
        "choices": [
            {
                "text": "",
                "logprobs": {
                    "content": [
                        {
                            "token": next(iter(top)),
                            "logprob": 0.0,
                            "top_logprobs": [
                                {"token": token, "logprob": lp} for token, lp in top.items()
                            ],
                        }
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1},
    }


TREE = {
    "name": "action",
    "question": {
        "type": "choice",
        "instructions": "What does the agent do?",
        "criteria": {"engage": "interact", "create": "write", "idle": "nothing"},
    },
    "children": {
        "engage": {
            "name": "engage_kind",
            "question": {
                "type": "choice",
                "instructions": "How to engage?",
                "criteria": {"like": "like", "repost": "repost", "none": "none"},
            },
        }
    },
}


def _client(table=None):
    fake = FakeCompletions(
        table
        or {
            "What does the agent do?": {
                "A": math.log(0.5),
                "B": math.log(0.3),
                "C": math.log(0.2),
            },
            "How to engage?": {"A": math.log(0.6), "B": math.log(0.3), "C": math.log(0.1)},
        }
    )
    backend = LocalReadoutBackend(base_url="http://fake/v1", model="m", post=fake)
    return SystemOneClient(backend), fake


def test_ask_tree_is_replayable():
    client, _ = _client()
    runs = [
        [
            (step.node, step.choice)
            for step in ask_tree(client, "state", TREE, random.Random(seed))
        ]
        for seed in (7, 7, 7)
    ]
    assert runs[0] == runs[1] == runs[2]

    paths = {
        tuple(step.choice for step in ask_tree(client, "state", TREE, random.Random(s)))
        for s in range(40)
    }
    # Sampling (not argmax) must reach more than one branch.
    assert len(paths) > 1


def test_ask_tree_appends_previous_answer_to_state():
    client, fake = _client(
        {
            "What does the agent do?": {"A": 0.0},
            "How to engage?": {"A": 0.0},
        }
    )
    path = ask_tree(client, "base-state", TREE, random.Random(1))
    assert [step.node for step in path] == ["action", "engage_kind"]
    assert [step.choice for step in path] == ["engage", "like"]
    assert "action: engage" in fake.prompts[-1]
    assert "action: engage" not in fake.prompts[0]


def test_ask_tree_rejects_trees_deeper_than_max_depth():
    import pytest

    client, _ = _client()
    with pytest.raises(ValueError):
        ask_tree(client, "s", TREE, random.Random(3), mode="argmax", max_depth=1)


def test_ask_tree_argmax_mode():
    client, _ = _client()
    path = ask_tree(client, "s", TREE, random.Random(3), mode="argmax")
    assert [step.choice for step in path] == ["engage", "like"]


def test_parse_top_logprobs_supports_vllm_and_llama_shapes():
    vllm = {"choices": [{"logprobs": {"top_logprobs": [{" A": -0.2, " B": -1.7}]}}]}
    llama = _llama_response({"A": -0.3, "B": -1.1})
    assert parse_top_logprobs(vllm) == {" A": -0.2, " B": -1.7}
    assert parse_top_logprobs(llama) == {"A": -0.3, "B": -1.1}


def test_local_backend_request_is_prefill_only_and_records_usage(tmp_path, monkeypatch):
    from app.utils.llm_usage import usage_stage

    fake = FakeCompletions({"Pick": {"A": -0.1, "B": -2.0}})
    backend = LocalReadoutBackend(base_url="http://fake/v1", model="m", post=fake, top_k=20)
    captured = []

    def post(payload):
        captured.append(payload)
        return fake(payload)

    backend._post = post
    request = SystemOneRequest(
        state="s",
        questions={"q": ChoiceQuestion(instructions="Pick", criteria={"x": "x", "y": "y"})},
    )
    with usage_stage("simulation", str(tmp_path)):
        response = SystemOneClient(backend).ask(request)
    assert captured[0]["max_tokens"] == 1
    assert captured[0]["logprobs"] == 20
    assert captured[0]["temperature"] == 0
    assert response.answers["q"].choice == "x"
    lines = (tmp_path / "llm_usage.jsonl").read_text(encoding="utf-8").splitlines()
    # Readout runs no decode step; it is recorded as zero decode tokens.
    assert json.loads(lines[0])["completion_tokens"] == 0
    assert json.loads(lines[0])["prompt_tokens"] == 10
