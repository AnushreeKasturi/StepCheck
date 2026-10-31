import pytest

from stepcheck.eval.benchmark import Outcome, Rate, compute_metrics, render_markdown, wilson


def record(i: int, error_line: int | None, error_type: str | None = None) -> dict[str, object]:
    return {"id": i, "family": "f", "lines": [], "error_line": error_line, "error_type": error_type}


def outcome(i: int, status: str, first: int | None, predicted: str | None = None) -> Outcome:
    return Outcome(i, status, first, predicted, (), 10.0 * (i + 1))


def test_wilson_interval_contains_estimate_and_handles_zero() -> None:
    lo, hi = wilson(5, 100)
    assert lo < 0.05 < hi
    lo, hi = wilson(0, 200)
    assert lo == 0.0 and 0.0 < hi < 0.03  # "0 false alarms" still has an upper bound


def test_rate_formatting() -> None:
    assert str(Rate(1, 4)).startswith("25.0% (")
    assert Rate(0, 0).value == 0.0


def test_metrics() -> None:
    records = [
        record(0, 2, "sign_error"),  # found, right type
        record(1, 3, "power_of_sum"),  # found, wrong type
        record(2, 3, "sign_error"),  # blamed line 2: too early
        record(3, None),  # correct, passed
        record(4, None),  # correct, false alarm
    ]
    outcomes = [
        outcome(0, "error", 2, "sign_error"),
        outcome(1, "error", 3, "arithmetic_slip"),
        outcome(2, "error", 2, "sign_error"),
        outcome(3, "correct", None),
        outcome(4, "error", 2, "unclassified"),
    ]
    m = compute_metrics(records, outcomes)
    assert (m["localization"].k, m["localization"].n) == (2, 3)
    assert (m["type_given_localized"].k, m["type_given_localized"].n) == (1, 2)
    assert m["end_to_end"].k == 1
    assert m["flagged_too_early"].k == 1
    assert (m["false_alarm"].k, m["false_alarm"].n) == (1, 2)
    assert m["confusion"]["power_of_sum"] == {"arithmetic_slip": 1}
    assert m["latency_ms"]["p50"] == pytest.approx(30.0)


def test_report_renders() -> None:
    m = compute_metrics([record(0, 2, "sign_error")], [outcome(0, "error", 2, "sign_error")])
    text = render_markdown(m, "cmd", workers=1)
    assert "Error localization" in text and "How to read these numbers" in text
