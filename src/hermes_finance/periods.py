"""Pure deterministic relative-period resolution (stage G3).

This module owns exactly one concern: mapping a canonical relative-period
token plus one explicit reference instant plus one business timezone to
the exact calendar period the finance queries need.

Supported concepts (deliberately only these four):

- ``today``: the local business date of the reference instant
- ``yesterday``: the local business date minus one calendar day
- ``current``: the local business calendar month
- ``previous``: the previous business calendar month

Architecture contract:

- pure functions with explicit inputs: the caller (the finance CLI
  boundary) supplies the reference instant and the business timezone;
  this module never reads the wall clock, never captures time itself,
  and performs no reads of any kind: no files, no variables of the
  host process, no network, no database, and no locale
- the reference instant MUST be timezone-aware; a naive datetime is
  rejected before any conversion happens
- the business timezone must be a usable ``datetime.tzinfo`` instance:
  the exported resolvers validate it themselves (``TypeError`` for a
  wrong type -- including ``None`` and IANA name strings -- and
  ``ValueError`` for an unusable or broken zone), so an invalid zone
  can never reach ``datetime.astimezone`` and silently resolve the
  period in the server-local timezone
- the single reference instant is converted into the supplied business
  timezone (an IANA ``zoneinfo.ZoneInfo`` constructed by the caller, or
  any other usable ``datetime.tzinfo``), and every period is resolved
  from that one conversion, so a zone whose UTC offset varies across
  the year (DST) is handled by the instant, not by the wall clock
- year boundaries are handled deterministically: January rolls back to
  December of the previous year, and a previous period that would cross
  before year 1 fails with ``ValueError`` instead of fabricating a year

The CLI boundary owns the clock (exactly one aware UTC snapshot per
relative invocation) and the IANA name parsing; this module owns the
arithmetic. That split keeps this layer fully deterministic and
independently testable.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, tzinfo
from typing import Final

from hermes_finance.domain import require_aware_datetime

__all__ = [
    "DATE_PERIODS",
    "MONTH_PERIODS",
    "resolve_relative_date",
    "resolve_relative_month",
]

#: The only supported relative date period tokens.
DATE_PERIODS: Final[frozenset[str]] = frozenset({"today", "yesterday"})

#: The only supported relative month period tokens.
MONTH_PERIODS: Final[frozenset[str]] = frozenset({"current", "previous"})

#: A fixed aware probe instant used to verify that a business timezone
#: is actually usable before any conversion happens. It is a constant,
#: never a clock read: its only job is to make the ``tzinfo`` answer one
#: offset question deterministically.
_USABILITY_PROBE: Final[datetime] = datetime(2000, 1, 2, 12, 0, 0, tzinfo=UTC)


def _require_usable_business_timezone(business_timezone: object) -> tzinfo:
    """Require a usable ``datetime.tzinfo`` business timezone.

    The exported resolvers form a public boundary of their own, so the
    business timezone is validated here and not only at the CLI edge:

    - anything that is not a ``datetime.tzinfo`` instance -- including
      ``None`` (which ``datetime.astimezone`` would otherwise silently
      interpret as the server-local timezone) and raw IANA name
      strings -- is rejected with ``TypeError``
    - a ``tzinfo`` that cannot answer an offset question for a fixed
      probe instant (``utcoffset`` returns ``None`` or a
      non-``timedelta`` value) is rejected with ``ValueError``: it is
      unusable for conversion
    - a ``tzinfo`` whose offset calculation raises is rejected with
      ``ValueError``, with the underlying exception chained as the
      cause

    No clock is read here: the probe instant is a module constant.
    """
    if not isinstance(business_timezone, tzinfo):
        raise TypeError(
            "business_timezone must be a datetime.tzinfo instance, got"
            f" {type(business_timezone).__name__!r}"
        )
    try:
        offset = business_timezone.utcoffset(
            _USABILITY_PROBE.replace(tzinfo=business_timezone)
        )
    # An arbitrary tzinfo subclass may raise literally anything from its
    # offset calculation; every such failure is normalised below.
    except Exception as error:
        raise ValueError(
            "business_timezone is broken: its UTC offset calculation"
            f" raised {type(error).__name__}"
        ) from error
    if not isinstance(offset, timedelta):
        # A zone answering with anything but a timedelta offset (None
        # included) cannot drive a conversion. That is a value defect of
        # the supplied zone, not a caller argument-type mistake, so the
        # deterministic rejection is ValueError per the public contract.
        raise ValueError(  # noqa: TRY004 - deliberate: unusable zone, not a type error
            "business_timezone is unusable: its UTC offset is not"
            " available for a fixed probe instant"
        )
    return business_timezone


def _local_business_date(
    reference_time: datetime, business_timezone: tzinfo
) -> date:
    """Convert one aware reference instant into the local business date.

    The reference instant must be timezone-aware and the business
    timezone must be a usable ``tzinfo``; both are validated here so an
    invalid timezone can never silently fall back to the server-local
    timezone. The instant is converted into the business timezone
    exactly once, and the local business date of that conversion is
    returned. No clock is read here: the caller owns the instant.

    The fixed usability probe cannot prove a zone works for every
    instant, so the actual conversion is protected too: if the supplied
    zone fails specifically for the real reference instant (a
    conditional or broken zone raising from, or answering unusably for,
    its offset calculation during ``astimezone``), that failure is
    normalized to a deterministic ``ValueError`` with the original
    exception preserved as the cause. No fallback timezone is ever
    attempted; the resolver fails closed.
    """
    aware = require_aware_datetime(reference_time, "reference_time")
    zone = _require_usable_business_timezone(business_timezone)
    # The try block deliberately surrounds ONLY the timezone conversion:
    # every exception escaping ``astimezone`` originates in the supplied
    # zone (an arbitrary ``tzinfo`` subclass may raise anything, and the
    # stdlib ``fromutc`` machinery raises on zones that cannot answer an
    # offset for this instant), so each one is a business-timezone
    # failure and is normalized below. No resolver or domain code runs
    # inside this block.
    try:
        localized = aware.astimezone(zone)
    except Exception as error:
        raise ValueError(
            "business_timezone failed during reference-time conversion:"
            f" {type(error).__name__}: {error}"
        ) from error
    return localized.date()


def resolve_relative_date(
    period: str,
    *,
    reference_time: datetime,
    business_timezone: tzinfo,
) -> date:
    """Resolve ``today`` / ``yesterday`` into an exact calendar date.

    ``period`` must be one of :data:`DATE_PERIODS`; any other token is
    rejected with ``ValueError``. ``reference_time`` must be
    timezone-aware and is the single instant the period is resolved
    against; ``business_timezone`` is the zone the business date is
    judged in.

    ``today`` is the local business date of the reference instant.
    ``yesterday`` is one calendar day earlier; if that would cross
    before year 1, a deterministic ``ValueError`` is raised instead of
    fabricating a date.
    """
    if period not in DATE_PERIODS:
        raise ValueError(
            f"unsupported relative date period: {period!r}"
            f" (expected one of {sorted(DATE_PERIODS)})"
        )
    local_date = _local_business_date(reference_time, business_timezone)
    if period == "today":
        return local_date
    previous_ordinal = local_date.toordinal() - 1
    if previous_ordinal < 1:
        raise ValueError(
            "yesterday of the minimum supported calendar date"
            " (0001-01-01) does not exist"
        )
    return date.fromordinal(previous_ordinal)


def resolve_relative_month(
    period: str,
    *,
    reference_time: datetime,
    business_timezone: tzinfo,
) -> tuple[int, int]:
    """Resolve ``current`` / ``previous`` into an exact ``(year, month)``.

    ``period`` must be one of :data:`MONTH_PERIODS`; any other token is
    rejected with ``ValueError``. ``reference_time`` must be
    timezone-aware and is the single instant the period is resolved
    against; ``business_timezone`` is the zone the business month is
    judged in.

    ``current`` is the local business calendar month of the reference
    instant. ``previous`` is one calendar month earlier: a normal month
    decrements the month, and January rolls back to December of the
    previous year. If that would cross before year 1, a deterministic
    ``ValueError`` is raised instead of fabricating a year.
    """
    if period not in MONTH_PERIODS:
        raise ValueError(
            f"unsupported relative month period: {period!r}"
            f" (expected one of {sorted(MONTH_PERIODS)})"
        )
    local_date = _local_business_date(reference_time, business_timezone)
    year, month = local_date.year, local_date.month
    if period == "current":
        return year, month
    if month == 1:
        if year <= 1:
            raise ValueError(
                "previous month of January of year 1 does not exist"
            )
        return year - 1, 12
    return year, month - 1
