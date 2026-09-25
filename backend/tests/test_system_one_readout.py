import math

import pytest

from app.system_one.models import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
)
from app.system_one.readout import (
    TooManyOptionsError,
    build_prompt,
    choice_labels,
    readout_answer,
    score_labels,
)


def _choice(n: int) -> ChoiceQuestion:
    return ChoiceQuestion(
        instructions="Pick one",
        criteria={f"opt{i}": f"option {i}" for i in range(n)},
    )


def test_choice_probabilities_sum_to_one():
    question = _choice(3)
    top = {"A": -0.1, "B": -2.5, "\n": -3.0}
    answer = readout_answer(question, top)

    assert set(answer.probabilities) == {"opt0", "opt1", "opt2"}
    assert math.isclose(sum(answer.probabilities.values()), 1.0, abs_tol=1e-6)
    assert answer.choice == "opt0"
    # C is missing from top-k, so it gets the floor and coverage < 1.
    assert answer.probabilities["opt2"] > 0
    assert answer.probabilities["opt2"] < answer.probabilities["opt1"]
    assert 0 < answer.coverage < 1
    assert answer.missing == ["opt2"]


def test_never_returns_unknown_option():
    question = _choice(3)
    top = {" D": -0.01, "A": -3.0, " B": -4.0, "C": -5.0, "\n": -1.0}
    answer = readout_answer(question, top)
    assert set(answer.probabilities) == {"opt0", "opt1", "opt2"}
    assert answer.choice == "opt0"


def test_more_than_26_options_raises():
    with pytest.raises(TooManyOptionsError):
        choice_labels(_choice(27))
    with pytest.raises(TooManyOptionsError):
        build_prompt("state", _choice(27))
    # 26 is allowed.
    assert choice_labels(_choice(26))[-1] == "Z"


def test_duplicate_label_tokens_keep_the_best_logprob():
    question = _choice(2)
    top = {" A": -2.0, "A": -0.5, "B": -1.0}
    answer = readout_answer(question, top)
    expected_a = math.exp(-0.5) / (math.exp(-0.5) + math.exp(-1.0))
    assert math.isclose(answer.probabilities["opt0"], expected_a, rel_tol=1e-9)


def test_score_expected_value_uses_zero_based_levels():
    question = ScoreQuestion(
        instructions="How angry",
        criteria=["calm", "annoyed", "furious"],
    )
    assert score_labels(question) == ["0", "1", "2"]
    top = {"0": math.log(0.2), "1": math.log(0.3), "2": math.log(0.5)}
    answer = readout_answer(question, top)
    assert math.isclose(sum(answer.probabilities.values()), 1.0, abs_tol=1e-6)
    assert math.isclose(answer.score, 0 * 0.2 + 1 * 0.3 + 2 * 0.5, rel_tol=1e-9)
    assert list(answer.probabilities) == ["calm", "annoyed", "furious"]


def test_score_level_limits():
    with pytest.raises(ValueError):
        ScoreQuestion(instructions="x", criteria=["only one"])
    with pytest.raises(ValueError):
        ScoreQuestion(instructions="x", criteria=[str(i) for i in range(11)])


def test_noul_compares_yes_and_no_only():
    question = NoulQuestion(instructions="Is it urgent?")
    top = {"Yes": math.log(0.6), "No": math.log(0.2), "Maybe": math.log(0.2)}
    answer = readout_answer(question, top)
    assert math.isclose(answer.noul, 0.75, rel_tol=1e-9)
    assert math.isclose(answer.coverage, 0.8, rel_tol=1e-9)


def test_temperature_flattens_distribution():
    question = _choice(2)
    top = {"A": -0.1, "B": -2.5}
    sharp = readout_answer(question, top, temperature=1.0)
    flat = readout_answer(question, top, temperature=3.0)
    assert flat.probabilities["opt0"] < sharp.probabilities["opt0"]
    assert flat.probabilities["opt0"] > 0.5


def test_all_labels_missing_gives_uniform_and_zero_coverage():
    question = _choice(4)
    answer = readout_answer(question, {"\n": -0.1, "The": -2.0})
    for value in answer.probabilities.values():
        assert math.isclose(value, 0.25, rel_tol=1e-9)
    assert answer.coverage == 0.0


def test_prompt_puts_state_before_question():
    state = "STATE-TEXT"
    prompt_a = build_prompt(state, _choice(3))
    prompt_b = build_prompt(state, NoulQuestion(instructions="Other question?"))
    assert prompt_a.index(state) < prompt_a.index("Pick one")
    # Same state -> identical prefix up to the end of the state block.
    cut = prompt_a.index(state) + len(state)
    assert prompt_a[:cut] == prompt_b[:cut]


def test_request_matches_jev_shape():
    request = SystemOneRequest.model_validate(
        {
            "state": "Help! My payouts have been failing for 3 days.",
            "model": "jev-latest",
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "criteria": {
                        "billing": "Payments, invoicing, refunds",
                        "technical": "Bugs, outages, integrations",
                    },
                },
                "anger": {
                    "type": "score",
                    "instructions": "How frustrated",
                    "criteria": ["calm", "frustrated", "angry"],
                },
                "urgent": {"type": "noul", "instructions": "Is it urgent?"},
            },
        }
    )
    assert isinstance(request.questions["department"], ChoiceQuestion)
    assert isinstance(request.questions["anger"], ScoreQuestion)
    assert isinstance(request.questions["urgent"], NoulQuestion)
    dumped = request.model_dump(exclude_none=True)
    assert dumped["questions"]["urgent"] == {"type": "noul", "instructions": "Is it urgent?"}
