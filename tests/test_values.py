import pytest

from circuitinsight.values import parse_value


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("10k", 1e4),
        ("1.5p", 1.5e-12),
        ("2meg", 2e6),
        ("800m", 0.8),
        ("100f", 1e-13),
        ("-0.5", -0.5),
        ("3", 3.0),
        ("4e-13", 4e-13),
        ("2.2n", 2.2e-9),
        ("1G", 1e9),
        ("10K", 1e4),
        ("10kohm", 1e4),
        (42, 42.0),
        (0.5, 0.5),
    ],
)
def test_parse(raw, expected):
    assert parse_value(raw) == pytest.approx(expected)


def test_reject_garbage():
    with pytest.raises(ValueError):
        parse_value("ten")
    with pytest.raises(ValueError):
        parse_value("1x")
