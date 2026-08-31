import pytest

from hedge_fund.run import validate_date_range


def test_validate_date_range_accepts_ordered_iso_dates():
    end, start = validate_date_range("2025-03-01", "2025-01-01")
    assert end.isoformat() == "2025-03-01"
    assert start.isoformat() == "2025-01-01"


def test_validate_date_range_rejects_bad_format():
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        validate_date_range("03/01/2025")


def test_validate_date_range_rejects_inverted_window():
    with pytest.raises(ValueError, match="on or before"):
        validate_date_range("2025-01-01", "2025-03-01")
