import pytest

from hedge_fund.run import (
    load_universe_inputs,
    package_version,
    validate_date_range,
    write_json_output,
)


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


def test_package_version_is_nonempty():
    assert package_version()


def test_write_json_output_creates_parents_and_replaces_atomically(tmp_path):
    destination = tmp_path / "nested" / "result.json"
    assert write_json_output(destination, '{"run": 1}') == destination
    assert destination.read_text() == '{"run": 1}\n'
    write_json_output(destination, '{"run": 2}')
    assert destination.read_text() == '{"run": 2}\n'
    assert list(destination.parent.iterdir()) == [destination]


def test_load_universe_inputs_combines_inline_file_and_comments(tmp_path):
    universe_file = tmp_path / "universe.txt"
    universe_file.write_text("msft, nvda # mega-cap tech\nBRK.B\n")
    assert load_universe_inputs("aapl, MSFT", universe_file) == [
        "AAPL", "MSFT", "NVDA", "BRK.B",
    ]


def test_load_universe_inputs_rejects_empty_sources(tmp_path):
    universe_file = tmp_path / "empty.txt"
    universe_file.write_text("# no symbols\n")
    with pytest.raises(ValueError, match="universe is empty"):
        load_universe_inputs(None, universe_file)
