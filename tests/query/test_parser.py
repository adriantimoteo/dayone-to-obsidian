from __future__ import annotations

import pytest
from datetime import date

import jkb.query.parser as _parser_module
from jkb.query.parser import ParsedQuery, parse_query


# ---------------------------------------------------------------------------
# Helpers — pin the "today" used by the parser to 2026-05-15 for determinism
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def pin_today(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_parser_module, "_TODAY", date(2026, 5, 15))


# ---------------------------------------------------------------------------
# ParsedQuery model defaults
# ---------------------------------------------------------------------------


def test_parsed_query_defaults() -> None:
    q = ParsedQuery()
    assert q.start_date is None
    assert q.end_date is None
    assert q.location is None
    assert q.tags == []
    assert q.keywords == ""


# ---------------------------------------------------------------------------
# Explicit year
# ---------------------------------------------------------------------------


def test_year_only() -> None:
    q = parse_query("entries from 2020")
    assert q.start_date == date(2020, 1, 1)
    assert q.end_date == date(2020, 12, 31)


def test_year_with_keyword() -> None:
    q = parse_query("entries from 2020 about travel")
    assert q.start_date == date(2020, 1, 1)
    assert q.end_date == date(2020, 12, 31)
    assert "travel" in q.keywords


def test_year_in_prefix() -> None:
    q = parse_query("in 2019 what did I write")
    assert q.start_date == date(2019, 1, 1)
    assert q.end_date == date(2019, 12, 31)


def test_year_during_prefix() -> None:
    q = parse_query("during 2021 trips")
    assert q.start_date == date(2021, 1, 1)
    assert q.end_date == date(2021, 12, 31)
    assert "trips" in q.keywords


def test_year_range() -> None:
    q = parse_query("from 2018 to 2020")
    assert q.start_date == date(2018, 1, 1)
    assert q.end_date == date(2020, 12, 31)


def test_year_range_between() -> None:
    q = parse_query("between 2015 and 2017 hiking")
    assert q.start_date == date(2015, 1, 1)
    assert q.end_date == date(2017, 12, 31)
    assert "hiking" in q.keywords


# ---------------------------------------------------------------------------
# Month + year
# ---------------------------------------------------------------------------


def test_month_year() -> None:
    q = parse_query("in March 2019")
    assert q.start_date == date(2019, 3, 1)
    assert q.end_date == date(2019, 3, 31)


def test_month_year_abbreviated() -> None:
    q = parse_query("Aug 2021 vacation")
    assert q.start_date == date(2021, 8, 1)
    assert q.end_date == date(2021, 8, 31)
    assert "vacation" in q.keywords


def test_month_year_during() -> None:
    q = parse_query("during October 2022")
    assert q.start_date == date(2022, 10, 1)
    assert q.end_date == date(2022, 10, 31)


def test_february_leap_year() -> None:
    q = parse_query("February 2020")
    assert q.start_date == date(2020, 2, 1)
    assert q.end_date == date(2020, 2, 29)


def test_february_non_leap_year() -> None:
    q = parse_query("February 2021")
    assert q.start_date == date(2021, 2, 1)
    assert q.end_date == date(2021, 2, 28)


# ---------------------------------------------------------------------------
# Relative dates (today = 2026-05-15)
# ---------------------------------------------------------------------------


def test_last_year() -> None:
    q = parse_query("what happened last year")
    assert q.start_date == date(2025, 1, 1)
    assert q.end_date == date(2025, 12, 31)


def test_this_year() -> None:
    q = parse_query("this year goals")
    assert q.start_date == date(2026, 1, 1)
    assert q.end_date == date(2026, 12, 31)


def test_next_year() -> None:
    q = parse_query("plans next year")
    assert q.start_date == date(2027, 1, 1)
    assert q.end_date == date(2027, 12, 31)


def test_last_month() -> None:
    q = parse_query("last month entries")
    assert q.start_date == date(2026, 4, 1)
    assert q.end_date == date(2026, 4, 30)


def test_this_month() -> None:
    q = parse_query("this month")
    assert q.start_date == date(2026, 5, 1)
    assert q.end_date == date(2026, 5, 31)


def test_yesterday() -> None:
    q = parse_query("yesterday")
    assert q.start_date == date(2026, 5, 14)
    assert q.end_date == date(2026, 5, 14)


def test_today() -> None:
    q = parse_query("today")
    assert q.start_date == date(2026, 5, 15)
    assert q.end_date == date(2026, 5, 15)


# ---------------------------------------------------------------------------
# Location extraction
# ---------------------------------------------------------------------------


def test_location_in() -> None:
    q = parse_query("what happened in Sagada last year")
    assert q.location == "Sagada"
    assert q.start_date == date(2025, 1, 1)
    assert q.end_date == date(2025, 12, 31)


def test_location_at() -> None:
    q = parse_query("memories at Tokyo")
    assert q.location == "Tokyo"


def test_location_multi_word() -> None:
    q = parse_query("entries in New York 2022")
    assert q.location == "New York"
    assert q.start_date == date(2022, 1, 1)


def test_no_location_lowercase() -> None:
    # Lowercase words after 'in' should not be picked up as locations
    q = parse_query("entries in 2020")
    assert q.location is None


# ---------------------------------------------------------------------------
# Tag extraction
# ---------------------------------------------------------------------------


def test_single_tag() -> None:
    q = parse_query("#travel 2020")
    assert "travel" in q.tags
    assert q.start_date == date(2020, 1, 1)


def test_multiple_tags() -> None:
    q = parse_query("#hiking #nature entries 2021")
    assert set(q.tags) == {"hiking", "nature"}


def test_tag_mixed_with_keyword() -> None:
    q = parse_query("camping #outdoors 2019")
    assert "outdoors" in q.tags
    assert "camping" in q.keywords


# ---------------------------------------------------------------------------
# Keywords (remaining text)
# ---------------------------------------------------------------------------


def test_keyword_passthrough() -> None:
    q = parse_query("surfing lessons")
    assert q.keywords == "surfing lessons"
    assert q.start_date is None
    assert q.location is None


def test_keywords_after_year_stripped() -> None:
    q = parse_query("entries from 2020 about travel")
    # "entries", "from 2020", "about" are consumed; "travel" stays
    assert "travel" in q.keywords


def test_no_filters() -> None:
    q = parse_query("random thoughts")
    assert q == ParsedQuery(keywords="random thoughts")


# ---------------------------------------------------------------------------
# Combined / ticket acceptance criteria
# ---------------------------------------------------------------------------


def test_acceptance_2020_travel() -> None:
    """'entries from 2020 about travel' → date range 2020 + keyword 'travel'."""
    q = parse_query("entries from 2020 about travel")
    assert q.start_date == date(2020, 1, 1)
    assert q.end_date == date(2020, 12, 31)
    assert "travel" in q.keywords


def test_acceptance_sagada_last_year() -> None:
    """'what happened in Sagada last year' → location Sagada + last year range."""
    q = parse_query("what happened in Sagada last year")
    assert q.location == "Sagada"
    assert q.start_date == date(2025, 1, 1)
    assert q.end_date == date(2025, 12, 31)


def test_empty_query() -> None:
    q = parse_query("")
    assert q == ParsedQuery()


def test_whitespace_only() -> None:
    q = parse_query("   ")
    assert q == ParsedQuery()
