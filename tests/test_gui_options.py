"""Unit tests for EmissionLWIR emcal-options parsing (no window is created)."""
import pytest

tk = pytest.importorskip("tkinter")
from speclab.EmissionLWIR import EmcalOptionsDialog  # noqa: E402

parse = EmcalOptionsDialog._parse_temp_spread


@pytest.mark.parametrize("text, method, expected", [
    ("", "nem", 0.0),
    ("0", "hullfit", 0.0),
    ("5", "alpha", 5.0),
    ("27", "graybody", 27.0),
    ("0, 50", "graybody", (0.0, 50.0)),
    ("0;50", "graybody", (0.0, 50.0)),
])
def test_valid(text, method, expected):
    assert parse(text, method) == expected


@pytest.mark.parametrize("text, method, match", [
    ("0, 50", "nem", "needs method 'graybody'"),
    ("5", "hullfit", "not supported for 'hullfit'"),
    ("5", "hullfit_linear", "not supported for 'hullfit_linear'"),
    ("-1", "nem", "must be ≥ 0"),
    ("5, 2", "graybody", "0 ≤ min < max"),
    ("1, 2, 3", "graybody", "one value or 'min, max'"),
    ("x", "nem", "could not convert"),
])
def test_invalid(text, method, match):
    with pytest.raises(ValueError, match=match):
        parse(text, method)
