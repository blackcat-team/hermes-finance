"""Tests for the pure relative-period resolver (stage G3).

All tests are deterministic: no network, no wall-clock time, no
randomness, no filesystem mutation, no database. Every resolution is
exercised with explicit ``reference_time`` instants and explicit
``tzinfo`` business timezones.

The finance test venv deliberately ships no IANA tz database (the
established stage-F5 test convention), so the fixed-offset business
zones used here are ``datetime.timezone`` instances with the exact
offsets the periods under test need, and the DST-capable case uses a
small deterministic ``tzinfo`` subclass with real transition behaviour.
The resolver accepts any usable ``tzinfo``, so no IANA lookup is
involved at this layer; IANA name parsing belongs to the CLI boundary
and is covered in ``test_cli.py``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Final

import pytest

import hermes_finance.periods as periods_module
from hermes_finance.periods import (
    resolve_relative_date,
    resolve_relative_month,
)

PERIODS_SOURCE: Final[str] = Path(periods_module.__file__).read_text(encoding="utf-8")

#: Moscow-equivalent business zone: fixed UTC+3 (no DST in the period tested).
MSK: Final[tzinfo] = timezone(timedelta(hours=3), "MSK")

#: A UTC+7 business zone used to show the zone choice changes the result.
ICT: Final[tzinfo] = timezone(timedelta(hours=7), "ICT")

#: US-Eastern-style fixed UTC-5 zone (standard time without DST).
EST_FIXED: Final[tzinfo] = timezone(timedelta(hours=-5), "EST")

#: Wall-clock bounds of the 2026 US Eastern daylight-saving window,
#: expressed unambiguously: 2026-03-08 03:00 is the first EDT wall time
#: and 2026-11-01 01:00 is the first EST wall time after the window.
#: (Built through an aware constant so the naive wall-clock value is
#: explicit and deliberate, following the established test convention.)
_DST_WALL_START: Final[datetime] = datetime(2026, 3, 8, 3, 0, tzinfo=UTC).replace(
    tzinfo=None
)
_DST_WALL_END: Final[datetime] = datetime(2026, 11, 1, 1, 0, tzinfo=UTC).replace(
    tzinfo=None
)

#: A deliberately naive reference instant for the rejection tests,
#: built the same way.
_NAIVE_REFERENCE: Final[datetime] = datetime(2026, 9, 5, 12, 0, tzinfo=UTC).replace(
    tzinfo=None
)


class DstEastern(tzinfo):
    """Deterministic DST-capable US-Eastern-style zone for 2026.

    EST (UTC-5) outside the summer window, EDT (UTC-4) inside it, with
    the 2026 US transition dates (second Sunday of March, first Sunday
    of November). The ambiguous and imaginary wall-clock hours around
    the transitions are never exercised by these tests.
    """

    def _is_dst(self, dt: datetime | None) -> bool:
        if dt is None:
            return False
        wall = dt.replace(tzinfo=None)
        return _DST_WALL_START <= wall < _DST_WALL_END

    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        if dt is None:
            return None
        return timedelta(hours=-4) if self._is_dst(dt) else timedelta(hours=-5)

    def dst(self, dt: datetime | None) -> timedelta | None:
        if dt is None:
            return None
        return timedelta(hours=1) if self._is_dst(dt) else timedelta(0)

    def tzname(self, dt: datetime | None) -> str | None:
        if dt is None:
            return None
        return "EDT" if self._is_dst(dt) else "EST"


#: The DST-capable zone under test.
EASTERN_DST: Final[tzinfo] = DstEastern()


# ---------------------------------------------------------------------------
# date periods: today / yesterday
# ---------------------------------------------------------------------------


def test_today_is_the_local_business_date() -> None:
    reference = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)
    assert resolve_relative_date(
        "today", reference_time=reference, business_timezone=MSK
    ) == date(2026, 9, 5)
    assert resolve_relative_date(
        "today", reference_time=reference, business_timezone=UTC
    ) == date(2026, 9, 5)


def test_yesterday_normal_day() -> None:
    reference = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)
    assert resolve_relative_date(
        "yesterday", reference_time=reference, business_timezone=MSK
    ) == date(2026, 9, 4)


def test_yesterday_month_boundary() -> None:
    # 12:00 UTC is 15:00 MSK on the first of the month.
    reference = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    assert resolve_relative_date(
        "yesterday", reference_time=reference, business_timezone=MSK
    ) == date(2026, 8, 31)


def test_yesterday_year_boundary() -> None:
    reference = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    assert resolve_relative_date(
        "yesterday", reference_time=reference, business_timezone=MSK
    ) == date(2025, 12, 31)


def test_yesterday_leap_year_boundary() -> None:
    reference = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)
    assert resolve_relative_date(
        "yesterday", reference_time=reference, business_timezone=UTC
    ) == date(2024, 2, 29)


def test_yesterday_of_minimum_date_fails_deterministically() -> None:
    reference = datetime(1, 1, 1, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="does not exist"):
        resolve_relative_date(
            "yesterday", reference_time=reference, business_timezone=UTC
        )


# ---------------------------------------------------------------------------
# month periods: current / previous
# ---------------------------------------------------------------------------


def test_current_month_is_the_local_business_month() -> None:
    reference = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)
    assert resolve_relative_month(
        "current", reference_time=reference, business_timezone=MSK
    ) == (2026, 9)


def test_previous_month_normal() -> None:
    reference = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)
    assert resolve_relative_month(
        "previous", reference_time=reference, business_timezone=MSK
    ) == (2026, 8)


def test_previous_month_january_rolls_to_previous_december() -> None:
    reference = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
    assert resolve_relative_month(
        "previous", reference_time=reference, business_timezone=MSK
    ) == (2025, 12)


def test_current_and_previous_in_december() -> None:
    reference = datetime(2025, 12, 10, 12, 0, tzinfo=UTC)
    assert resolve_relative_month(
        "current", reference_time=reference, business_timezone=MSK
    ) == (2025, 12)
    assert resolve_relative_month(
        "previous", reference_time=reference, business_timezone=MSK
    ) == (2025, 11)


def test_previous_month_of_january_year_1_fails_deterministically() -> None:
    reference = datetime(1, 1, 15, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="does not exist"):
        resolve_relative_month(
            "previous", reference_time=reference, business_timezone=UTC
        )


# ---------------------------------------------------------------------------
# aware reference requirement
# ---------------------------------------------------------------------------


def test_naive_reference_time_rejected_for_date_resolution() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        resolve_relative_date(
            "today", reference_time=_NAIVE_REFERENCE, business_timezone=MSK
        )


def test_naive_reference_time_rejected_for_month_resolution() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        resolve_relative_month(
            "current", reference_time=_NAIVE_REFERENCE, business_timezone=MSK
        )


def test_non_datetime_reference_time_rejected() -> None:
    with pytest.raises(TypeError):
        resolve_relative_date(
            "today", reference_time="2026-09-05T12:00:00+03:00", business_timezone=MSK  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# the business timezone actually matters
# ---------------------------------------------------------------------------


def test_business_timezone_changes_the_resolved_date() -> None:
    """21:30 UTC is already the next day in UTC+3 but not in UTC."""
    reference = datetime(2026, 9, 5, 21, 30, tzinfo=UTC)
    assert resolve_relative_date(
        "today", reference_time=reference, business_timezone=MSK
    ) == date(2026, 9, 6)
    assert resolve_relative_date(
        "today", reference_time=reference, business_timezone=UTC
    ) == date(2026, 9, 5)
    assert resolve_relative_date(
        "today", reference_time=reference, business_timezone=ICT
    ) == date(2026, 9, 6)


def test_near_midnight_business_zone_boundary_yesterday() -> None:
    reference = datetime(2026, 9, 5, 21, 30, tzinfo=UTC)
    assert resolve_relative_date(
        "yesterday", reference_time=reference, business_timezone=MSK
    ) == date(2026, 9, 5)
    # The same instant one second before the Moscow midnight boundary.
    before = datetime(2026, 9, 5, 20, 59, 59, tzinfo=UTC)
    assert resolve_relative_date(
        "today", reference_time=before, business_timezone=MSK
    ) == date(2026, 9, 5)


def test_near_midnight_business_zone_boundary_month() -> None:
    # 2026-10-01 00:30 MSK is still 2026-09-30 in UTC.
    reference = datetime(2026, 9, 30, 21, 30, tzinfo=UTC)
    assert resolve_relative_month(
        "current", reference_time=reference, business_timezone=MSK
    ) == (2026, 10)
    assert resolve_relative_month(
        "current", reference_time=reference, business_timezone=UTC
    ) == (2026, 9)


def test_dst_capable_zone_resolves_by_instant_offset() -> None:
    """04:30 UTC on July 1 is July 1 under EDT but June 30 under EST."""
    summer = datetime(2026, 7, 1, 4, 30, tzinfo=UTC)
    assert resolve_relative_date(
        "today", reference_time=summer, business_timezone=EASTERN_DST
    ) == date(2026, 7, 1)
    # A fixed UTC-5 zone without DST resolves the same instant to the
    # previous local date: the DST offset at the instant is what counts.
    assert resolve_relative_date(
        "today", reference_time=summer, business_timezone=EST_FIXED
    ) == date(2026, 6, 30)


def test_dst_capable_zone_winter_instant() -> None:
    """The same wall-clock UTC time in winter resolves a day earlier."""
    winter = datetime(2026, 1, 15, 4, 30, tzinfo=UTC)
    assert resolve_relative_date(
        "today", reference_time=winter, business_timezone=EASTERN_DST
    ) == date(2026, 1, 14)


def test_dst_capable_zone_yesterday_across_spring_forward() -> None:
    # 2026-03-09 04:30 UTC is 00:30 EDT on March 9 (DST began March 8).
    reference = datetime(2026, 3, 9, 4, 30, tzinfo=UTC)
    assert resolve_relative_date(
        "yesterday", reference_time=reference, business_timezone=EASTERN_DST
    ) == date(2026, 3, 8)


# ---------------------------------------------------------------------------
# invalid period tokens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_period", ["tomorrow", "current", "TODAY", "today ", "", "последние 7 дней"]
)
def test_invalid_date_period_token_rejected(bad_period: str) -> None:
    reference = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="unsupported relative date period"):
        resolve_relative_date(
            bad_period, reference_time=reference, business_timezone=MSK
        )


@pytest.mark.parametrize(
    "bad_period", ["next", "today", "CURRENT", "current ", "", "этот год"]
)
def test_invalid_month_period_token_rejected(bad_period: str) -> None:
    reference = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="unsupported relative month period"):
        resolve_relative_month(
            bad_period, reference_time=reference, business_timezone=MSK
        )


# ---------------------------------------------------------------------------
# business timezone validation (the exported resolvers are a public
# boundary and must be safe independently of CLI validation)
# ---------------------------------------------------------------------------


class NoneOffsetZone(tzinfo):
    """A ``tzinfo`` whose offset is never available (``utcoffset`` -> None)."""

    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        return None

    def dst(self, dt: datetime | None) -> timedelta | None:
        return None

    def tzname(self, dt: datetime | None) -> str | None:
        return "Unusable"


class RaisingZone(tzinfo):
    """A ``tzinfo`` whose offset calculation always raises."""

    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        raise RuntimeError("offset calculation exploded")

    def dst(self, dt: datetime | None) -> timedelta | None:
        raise RuntimeError("offset calculation exploded")

    def tzname(self, dt: datetime | None) -> str | None:
        return "Broken"


#: The fixed wall-clock instant the resolver's usability probe uses
#: (imported from the module under test so the doubles below always
#: agree with the actual probe).
_PROBE_WALL: Final[datetime] = periods_module._USABILITY_PROBE.replace(tzinfo=None)


class ProbePassingConversionRaisingZone(tzinfo):
    """Passes the fixed usability probe, raises at every other instant.

    Answers a perfectly valid fixed offset for the resolver's probe
    instant, so pre-validation accepts it, but raises ``RuntimeError``
    from ``utcoffset`` for the actual reference-time conversion. This
    is the exact production defect double: the failure happens only
    during ``astimezone`` for the real supplied instant.
    """

    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        if dt is None:
            return None
        if dt.replace(tzinfo=None) == _PROBE_WALL:
            return timedelta(hours=3)
        raise RuntimeError("offset calculation exploded for this instant")

    def dst(self, dt: datetime | None) -> timedelta | None:
        return timedelta(0)

    def tzname(self, dt: datetime | None) -> str | None:
        return "Conditional"


class ProbePassingNoneOffsetZone(tzinfo):
    """Passes the fixed usability probe, answers None at other instants.

    Answers a valid offset for the probe instant but ``None`` from
    ``utcoffset`` for the actual conversion instant, which the stdlib
    ``fromutc`` machinery rejects with a confusing internal message.
    The resolver must normalize that failure too.
    """

    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        if dt is None:
            return None
        if dt.replace(tzinfo=None) == _PROBE_WALL:
            return timedelta(hours=3)
        return None

    def dst(self, dt: datetime | None) -> timedelta | None:
        return timedelta(0)

    def tzname(self, dt: datetime | None) -> str | None:
        return "ConditionalNone"


#: A fixed aware reference instant for the validation tests: 21:30 UTC
#: on 2026-09-05, which is already 2026-09-06 in a UTC+3 zone.
_VALIDATION_REFERENCE: Final[datetime] = datetime(2026, 9, 5, 21, 30, tzinfo=UTC)

#: Inputs that are not a ``tzinfo`` instance at all: ``None`` (the
#: dangerous case: ``astimezone(None)`` would silently select the
#: server-local timezone), a raw IANA name string, an int, an arbitrary
#: object, and a container.
_INVALID_ZONE_TYPES: Final[list[object]] = [
    None,
    "Europe/Moscow",
    3,
    object(),
    ["Europe/Moscow"],
]


@pytest.mark.parametrize("bad_zone", _INVALID_ZONE_TYPES)
def test_non_tzinfo_business_timezone_rejected_for_date_resolution(
    bad_zone: object,
) -> None:
    with pytest.raises(TypeError, match="business_timezone must be a datetime.tzinfo"):
        resolve_relative_date(
            "today",
            reference_time=_VALIDATION_REFERENCE,
            business_timezone=bad_zone,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad_zone", _INVALID_ZONE_TYPES)
def test_non_tzinfo_business_timezone_rejected_for_month_resolution(
    bad_zone: object,
) -> None:
    with pytest.raises(TypeError, match="business_timezone must be a datetime.tzinfo"):
        resolve_relative_month(
            "current",
            reference_time=_VALIDATION_REFERENCE,
            business_timezone=bad_zone,  # type: ignore[arg-type]
        )


def test_none_business_timezone_never_silently_falls_back_to_server_local() -> None:
    """The headline regression: ``None`` must never mean "local time".

    Before the fix, ``business_timezone=None`` reached
    ``datetime.astimezone(None)``, which silently resolved the period
    in the host/server-local timezone. It must instead be rejected
    deterministically, for both exported resolvers.
    """
    with pytest.raises(TypeError, match="business_timezone must be a datetime.tzinfo"):
        resolve_relative_date(
            "today",
            reference_time=_VALIDATION_REFERENCE,
            business_timezone=None,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="business_timezone must be a datetime.tzinfo"):
        resolve_relative_month(
            "current",
            reference_time=_VALIDATION_REFERENCE,
            business_timezone=None,  # type: ignore[arg-type]
        )


def test_unusable_none_offset_zone_rejected_for_date_resolution() -> None:
    with pytest.raises(ValueError, match="business_timezone is unusable"):
        resolve_relative_date(
            "today",
            reference_time=_VALIDATION_REFERENCE,
            business_timezone=NoneOffsetZone(),
        )


def test_unusable_none_offset_zone_rejected_for_month_resolution() -> None:
    with pytest.raises(ValueError, match="business_timezone is unusable"):
        resolve_relative_month(
            "current",
            reference_time=_VALIDATION_REFERENCE,
            business_timezone=NoneOffsetZone(),
        )


def test_broken_raising_zone_rejected_deterministically_for_date_resolution() -> None:
    with pytest.raises(ValueError, match="business_timezone is broken") as excinfo:
        resolve_relative_date(
            "today",
            reference_time=_VALIDATION_REFERENCE,
            business_timezone=RaisingZone(),
        )
    # The underlying zone failure is preserved as the cause.
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_broken_raising_zone_rejected_deterministically_for_month_resolution() -> None:
    with pytest.raises(ValueError, match="business_timezone is broken") as excinfo:
        resolve_relative_month(
            "current",
            reference_time=_VALIDATION_REFERENCE,
            business_timezone=RaisingZone(),
        )
    assert isinstance(excinfo.value.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# actual-conversion failure contract: a zone may pass the fixed probe
# but fail for the real reference instant
# ---------------------------------------------------------------------------


def test_probe_passing_conversion_raising_zone_fails_closed_for_date_resolution() -> None:
    """A zone failing only at the real instant must fail closed.

    The double passes the fixed usability probe (valid offset at the
    probe instant) but raises ``RuntimeError`` during the actual
    ``astimezone`` conversion of the supplied reference instant. The
    resolver must surface a deterministic ``ValueError`` -- never the
    raw ``RuntimeError`` -- with the original exception preserved as
    ``__cause__``.
    """
    zone = ProbePassingConversionRaisingZone()
    with pytest.raises(
        ValueError, match="failed during reference-time conversion"
    ) as excinfo:
        resolve_relative_date(
            "today",
            reference_time=_VALIDATION_REFERENCE,
            business_timezone=zone,
        )
    # The original zone failure is preserved as the cause.
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "exploded for this instant" in str(excinfo.value.__cause__)


def test_probe_passing_conversion_raising_zone_fails_closed_for_month_resolution() -> None:
    with pytest.raises(
        ValueError, match="failed during reference-time conversion"
    ) as excinfo:
        resolve_relative_month(
            "current",
            reference_time=_VALIDATION_REFERENCE,
            business_timezone=ProbePassingConversionRaisingZone(),
        )
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_probe_passing_none_offset_zone_fails_closed_for_date_resolution() -> None:
    """A zone answering None at the real instant must fail closed too.

    The double passes the probe but answers ``None`` for the actual
    conversion instant, which the stdlib ``fromutc`` machinery rejects
    internally. That internal failure must be normalized to the same
    deterministic ``ValueError`` with the cause preserved, not leaked
    with the stdlib's confusing internal wording.
    """
    with pytest.raises(
        ValueError, match="failed during reference-time conversion"
    ) as excinfo:
        resolve_relative_date(
            "today",
            reference_time=_VALIDATION_REFERENCE,
            business_timezone=ProbePassingNoneOffsetZone(),
        )
    assert excinfo.value.__cause__ is not None


def test_probe_passing_none_offset_zone_fails_closed_for_month_resolution() -> None:
    with pytest.raises(
        ValueError, match="failed during reference-time conversion"
    ) as excinfo:
        resolve_relative_month(
            "current",
            reference_time=_VALIDATION_REFERENCE,
            business_timezone=ProbePassingNoneOffsetZone(),
        )
    assert excinfo.value.__cause__ is not None


def test_conversion_failure_never_silently_uses_another_timezone() -> None:
    """A conversion failure surfaces as the failure itself, never a date.

    If the resolver silently retried in another timezone (UTC, local,
    or a fallback), the call would return a date instead of raising.
    Both resolvers must instead propagate the normalized failure for
    every period token, so all four tokens are pinned to fail closed.
    """
    for period in ("today", "yesterday"):
        with pytest.raises(ValueError, match="failed during reference-time"):
            resolve_relative_date(
                period,
                reference_time=_VALIDATION_REFERENCE,
                business_timezone=ProbePassingConversionRaisingZone(),
            )
    for period in ("current", "previous"):
        with pytest.raises(ValueError, match="failed during reference-time"):
            resolve_relative_month(
                period,
                reference_time=_VALIDATION_REFERENCE,
                business_timezone=ProbePassingConversionRaisingZone(),
            )


def test_valid_utc_business_timezone_accepted_by_both_resolvers() -> None:
    assert resolve_relative_date(
        "today", reference_time=_VALIDATION_REFERENCE, business_timezone=UTC
    ) == date(2026, 9, 5)
    assert resolve_relative_date(
        "yesterday", reference_time=_VALIDATION_REFERENCE, business_timezone=UTC
    ) == date(2026, 9, 4)
    assert resolve_relative_month(
        "current", reference_time=_VALIDATION_REFERENCE, business_timezone=UTC
    ) == (2026, 9)
    assert resolve_relative_month(
        "previous", reference_time=_VALIDATION_REFERENCE, business_timezone=UTC
    ) == (2026, 8)


def test_valid_fixed_offset_business_timezone_accepted_by_both_resolvers() -> None:
    assert resolve_relative_date(
        "today", reference_time=_VALIDATION_REFERENCE, business_timezone=MSK
    ) == date(2026, 9, 6)
    assert resolve_relative_month(
        "previous", reference_time=_VALIDATION_REFERENCE, business_timezone=MSK
    ) == (2026, 8)


def test_valid_dst_capable_business_timezone_accepted_by_both_resolvers() -> None:
    """The existing DST-capable test zone passes validation and resolves."""
    summer = datetime(2026, 7, 1, 4, 30, tzinfo=UTC)
    assert resolve_relative_date(
        "today", reference_time=summer, business_timezone=EASTERN_DST
    ) == date(2026, 7, 1)
    assert resolve_relative_month(
        "current", reference_time=summer, business_timezone=EASTERN_DST
    ) == (2026, 7)


def test_valid_resolutions_unchanged_after_timezone_validation() -> None:
    """No behavior change for valid inputs, including boundaries.

    Pins the near-midnight business-zone boundary and repeat
    determinism after the validation fix: the same instant resolves the
    same way every time.
    """
    # 21:30 UTC is 2026-09-06 in Moscow but still 2026-09-05 in UTC.
    assert resolve_relative_date(
        "today", reference_time=_VALIDATION_REFERENCE, business_timezone=MSK
    ) == date(2026, 9, 6)
    assert resolve_relative_date(
        "yesterday", reference_time=_VALIDATION_REFERENCE, business_timezone=MSK
    ) == date(2026, 9, 5)
    assert resolve_relative_date(
        "today", reference_time=_VALIDATION_REFERENCE, business_timezone=UTC
    ) == date(2026, 9, 5)
    assert resolve_relative_month(
        "current", reference_time=_VALIDATION_REFERENCE, business_timezone=MSK
    ) == (2026, 9)
    assert resolve_relative_month(
        "previous", reference_time=_VALIDATION_REFERENCE, business_timezone=MSK
    ) == (2026, 8)
    # Repeated resolution of the same instant is deterministic.
    assert resolve_relative_date(
        "today", reference_time=_VALIDATION_REFERENCE, business_timezone=MSK
    ) == date(2026, 9, 6)


# ---------------------------------------------------------------------------
# determinism and representation independence
# ---------------------------------------------------------------------------


def test_deterministic_repeat_result() -> None:
    reference = datetime(2026, 9, 5, 21, 30, tzinfo=UTC)
    first_date = resolve_relative_date(
        "today", reference_time=reference, business_timezone=MSK
    )
    second_date = resolve_relative_date(
        "today", reference_time=reference, business_timezone=MSK
    )
    assert first_date == second_date == date(2026, 9, 6)
    first_month = resolve_relative_month(
        "previous", reference_time=reference, business_timezone=MSK
    )
    second_month = resolve_relative_month(
        "previous", reference_time=reference, business_timezone=MSK
    )
    assert first_month == second_month == (2026, 8)


def test_resolution_depends_on_the_instant_not_its_representation() -> None:
    """The same instant in two aware representations resolves identically.

    There is no dependence on any server-local timezone: only the
    explicit reference instant and the explicit business timezone
    participate in the resolution.
    """
    as_utc = datetime(2026, 9, 5, 21, 30, tzinfo=UTC)
    as_plus3 = datetime(2026, 9, 6, 0, 30, tzinfo=timezone(timedelta(hours=3)))
    assert as_utc == as_plus3
    assert resolve_relative_date(
        "today", reference_time=as_utc, business_timezone=MSK
    ) == resolve_relative_date(
        "today", reference_time=as_plus3, business_timezone=MSK
    )


# ---------------------------------------------------------------------------
# purity: no clock, no I/O in the resolver module
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    [
        "datetime.now",
        "utcnow",
        "date.today",
        "time.time",
        "time.sleep",
        "monotonic",
        "perf_counter",
        "astimezone(None)",
        "environ",
        "import os",
        "open(",
        "Path(",
        "sqlite3",
        "subprocess",
        "socket",
        "urllib",
        "requests",
        "setlocale",
    ],
)
def test_periods_source_has_no_clock_or_io_access(forbidden: str) -> None:
    assert forbidden not in PERIODS_SOURCE


def test_periods_module_exports_only_the_resolver_api() -> None:
    assert sorted(periods_module.__all__) == [
        "DATE_PERIODS",
        "MONTH_PERIODS",
        "resolve_relative_date",
        "resolve_relative_month",
    ]
    assert periods_module.DATE_PERIODS == frozenset({"today", "yesterday"})
    assert periods_module.MONTH_PERIODS == frozenset({"current", "previous"})
