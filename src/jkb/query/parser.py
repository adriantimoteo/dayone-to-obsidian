from __future__ import annotations

import re
from datetime import date, timedelta
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ParsedQuery(BaseModel):
    start_date: date | None = None
    end_date: date | None = None
    location: str | None = None
    tags: list[str] = []
    keywords: str = ""


# ---------------------------------------------------------------------------
# Reference date (injected for testing; defaults to today)
# ---------------------------------------------------------------------------

_TODAY: date | None = None


def _today() -> date:
    return _TODAY if _TODAY is not None else date.today()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Patterns are applied left-to-right; matched spans are consumed from the
# working copy of the query so they don't bleed into keywords.

_YEAR_FULL = r"\b(1[89]\d{2}|20\d{2})\b"
_MONTH_NAMES = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_MONTH_YEAR = rf"\b({_MONTH_NAMES})\s+({_YEAR_FULL[2:-2]})\b"
_MONTH_ONLY = rf"\b({_MONTH_NAMES})\b"

_MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_LOCATION_PREPS = r"\b(?:in|at|near|from)\b"
# A location token is a Title-Case word (or a run of them) that is not a
# common English stop-word and not a known keyword we handle elsewhere.
_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "for", "with",
    "on", "about", "last", "this", "next", "year", "month", "week",
    "today", "yesterday", "entries", "what", "happened", "journal",
}


def _month_range(month: int, year: int) -> tuple[date, date]:
    """Return (first, last) day of the given month/year."""
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last_day)


def _extract_relative_date(text: str) -> tuple[date | None, date | None, str]:
    """Handle 'last year', 'this year', 'last month', 'this month', etc."""
    today = _today()
    start: date | None = None
    end: date | None = None

    patterns = [
        (r"\blast\s+year\b", "last_year"),
        (r"\bthis\s+year\b", "this_year"),
        (r"\bnext\s+year\b", "next_year"),
        (r"\blast\s+month\b", "last_month"),
        (r"\bthis\s+month\b", "this_month"),
        (r"\bnext\s+month\b", "next_month"),
        (r"\bthis\s+week\b", "this_week"),
        (r"\blast\s+week\b", "last_week"),
        (r"\byesterday\b", "yesterday"),
        (r"\btoday\b", "today"),
    ]

    for pattern, key in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            text = text[: m.start()] + text[m.end() :]
            if key == "last_year":
                start = date(today.year - 1, 1, 1)
                end = date(today.year - 1, 12, 31)
            elif key == "this_year":
                start = date(today.year, 1, 1)
                end = date(today.year, 12, 31)
            elif key == "next_year":
                start = date(today.year + 1, 1, 1)
                end = date(today.year + 1, 12, 31)
            elif key == "last_month":
                first = today.replace(day=1)
                prev = date(first.year, first.month - 1, 1) if first.month > 1 else date(first.year - 1, 12, 1)
                start, end = _month_range(prev.month, prev.year)
            elif key == "this_month":
                start, end = _month_range(today.month, today.year)
            elif key == "next_month":
                if today.month == 12:
                    start, end = _month_range(1, today.year + 1)
                else:
                    start, end = _month_range(today.month + 1, today.year)
            elif key == "this_week":
                start = today - timedelta(days=today.weekday())
                end = start + timedelta(days=6)
            elif key == "last_week":
                start = today - timedelta(days=today.weekday() + 7)
                end = start + timedelta(days=6)
            elif key == "yesterday":
                start = end = today - timedelta(days=1)
            elif key == "today":
                start = end = today
            break

    return start, end, text


def _extract_year_range(text: str) -> tuple[date | None, date | None, str]:
    """Handle explicit 4-digit years: 'in 2020', 'from 2018 to 2020', 'entries from 2020'."""
    # Range: "from YYYY to YYYY" / "between YYYY and YYYY"
    range_pat = re.compile(
        r"\b(?:from|between)\s+" + _YEAR_FULL[2:-2] + r"\s+(?:to|and)\s+" + _YEAR_FULL[2:-2] + r"\b",
        re.IGNORECASE,
    )
    m = range_pat.search(text)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        if y1 > y2:
            y1, y2 = y2, y1
        text = text[: m.start()] + text[m.end() :]
        return date(y1, 1, 1), date(y2, 12, 31), text

    # Single year with optional "in/from/during"
    single_pat = re.compile(
        r"\b(?:in|from|during|of)?\s*(" + _YEAR_FULL[2:-2] + r")\b",
        re.IGNORECASE,
    )
    m = single_pat.search(text)
    if m:
        y = int(m.group(1))
        text = text[: m.start()] + text[m.end() :]
        return date(y, 1, 1), date(y, 12, 31), text

    return None, None, text


def _extract_month_year(text: str) -> tuple[date | None, date | None, str]:
    """Handle 'in March 2019', 'March 2019', 'during August 2021'."""
    pat = re.compile(
        r"\b(?:in|during|on)?\s*(" + _MONTH_NAMES[2:-1] + r")\s+(\d{4})\b",
        re.IGNORECASE,
    )
    m = pat.search(text)
    if m:
        month = _MONTH_MAP[m.group(1).lower()[:3]]
        year = int(m.group(2))
        start, end = _month_range(month, year)
        text = text[: m.start()] + text[m.end() :]
        return start, end, text
    return None, None, text


def _extract_tags(text: str) -> tuple[list[str], str]:
    """Extract explicit #tag tokens."""
    tags = re.findall(r"#(\w+)", text)
    text = re.sub(r"#\w+", "", text)
    return tags, text


def _extract_location(text: str) -> tuple[str | None, str]:
    """Extract a location following a preposition: 'in Sagada', 'at Tokyo'."""
    # Match a prep followed by one or more Title-Case words
    pat = re.compile(
        r"\b(?:in|at|near)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b"
    )
    m = pat.search(text)
    if m:
        candidate = m.group(1)
        # Reject if it looks like a year phrase or known stop-word
        if candidate.lower() not in _STOP_WORDS and not re.fullmatch(r"\d{4}", candidate):
            text = text[: m.start()] + text[m.end() :]
            return candidate, text
    return None, text


def _normalise_whitespace(text: str) -> str:
    # Remove common filler phrases left over after extraction
    fillers = [
        r"\bentries\b",
        r"\bentry\b",
        r"\bwhat\s+happened\b",
        r"\bshow\s+me\b",
        r"\bfind\b",
        r"\babout\b",
        r"\bthe\b",
    ]
    for filler in fillers:
        text = re.sub(filler, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip(" ,;")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_query(query: str) -> ParsedQuery:
    """Parse a natural-language query string into structured filters.

    Extraction order matters: relative dates before absolute years so that
    'last year' does not get its year digit captured as a standalone year.
    """
    text = query.strip()

    tags, text = _extract_tags(text)

    start, end, text = _extract_relative_date(text)

    if start is None:
        start, end, text = _extract_month_year(text)

    if start is None:
        start, end, text = _extract_year_range(text)

    location, text = _extract_location(text)

    keywords = _normalise_whitespace(text)

    return ParsedQuery(
        start_date=start,
        end_date=end,
        location=location,
        tags=tags,
        keywords=keywords,
    )
