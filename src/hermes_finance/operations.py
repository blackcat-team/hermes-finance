"""Read-only Finance query operations for Hermes Finance (stage D1).

This module is the deliberate operations layer between Hermes (the
future Telegram-facing component) and the accepted repository
primitives. It exposes exactly three small function-based queries:

- :func:`list_recent_transactions`: the newest ACTIVE transactions
  ("last N")
- :func:`list_transactions_by_date`: the transactions of one exact
  caller-supplied business date
- :func:`list_transactions_by_month`: the transactions of one exact
  calendar month

Contract highlights:

- every query returns only ``TransactionStatus.ACTIVE`` rows:
  soft-deleted transactions are invisible to normal user queries, and
  there is deliberately no ``include_deleted`` escape hatch in D1
- every query returns immutable :class:`~hermes_finance.domain.Transaction`
  objects reconstructed through the accepted C2 repository row mapper;
  there is no second mapping path and no provenance DTO
- ordering is deterministic newest-first:
  ``transaction_date DESC, id DESC``; ``created_at`` is never an
  ordering authority
- every query is strictly read-only: no rows are written, no
  transaction is started, committed, or rolled back, no pragma is
  touched, no migration runs, no second connection is opened, and a
  caller-owned active transaction is left untouched
- the wall clock is never consulted: there is deliberately no
  "today" operation; the caller always supplies the business date or
  calendar month explicitly (Hermes will later derive them from
  ``FinanceConfig.business_timezone``)
- no aggregation is performed: totals, nets, and summaries belong to
  the stage-E reporting core
- input validation is deterministic and happens before any SQL
  execution; there is no silent coercion of bools, floats, or strings
"""

from __future__ import annotations

import sqlite3
from datetime import date
from typing import Final

from hermes_finance.domain import Transaction
from hermes_finance.repository import (
    list_active_transactions_by_date,
    list_active_transactions_in_month,
    list_active_transactions_recent,
)

__all__ = [
    "list_recent_transactions",
    "list_transactions_by_date",
    "list_transactions_by_month",
]

#: Default and bounds of the ``limit`` contract for recent queries.
DEFAULT_LIMIT: Final[int] = 10
MIN_LIMIT: Final[int] = 1
MAX_LIMIT: Final[int] = 100


def _require_limit(value: object) -> int:
    """Validate the recent-query ``limit`` contract.

    ``limit`` must be a real ``int`` (``bool`` is explicitly rejected,
    even though ``bool`` is an ``int`` subclass) between 1 and 100
    inclusive. Floats, strings, ``None``, and out-of-range values are
    rejected deterministically before any SQL execution; nothing is
    ever coerced.
    """
    if isinstance(value, bool):
        raise TypeError("limit must not be a bool")
    if not isinstance(value, int):
        raise TypeError(f"limit must be an int, got {type(value).__name__!r}")
    if not MIN_LIMIT <= value <= MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and 100, got {value!r}")
    return value


def list_recent_transactions(
    connection: sqlite3.Connection,
    *,
    limit: int = DEFAULT_LIMIT,
) -> tuple[Transaction, ...]:
    """Return up to ``limit`` newest ACTIVE transactions ("last N").

    ``connection`` must be a real caller-owned
    :class:`sqlite3.Connection` prepared by
    :func:`hermes_finance.storage.open_database`; lookalike objects
    are rejected cleanly with a :class:`TypeError`.

    ``limit`` defaults to 10 and must be a real ``int`` between 1 and
    100 inclusive; ``bool``, ``float``, ``str``, ``None``, zero,
    negative, and greater-than-100 values are rejected before any SQL
    execution.

    Results contain only ``TransactionStatus.ACTIVE`` rows (deleted
    transactions are invisible) ordered newest-first by
    ``transaction_date DESC, id DESC``. Returns ``()`` for an empty
    ledger.

    Read-only: no rows are written and no transaction is started,
    committed, or rolled back.
    """
    validated_limit = _require_limit(limit)
    return list_active_transactions_recent(connection, limit=validated_limit)


def list_transactions_by_date(
    connection: sqlite3.Connection,
    transaction_date: date,
) -> tuple[Transaction, ...]:
    """Return the ACTIVE transactions of one exact business date.

    ``transaction_date`` must be a plain ``datetime.date`` supplied
    explicitly by the caller; ``datetime`` instances and arbitrary
    non-date values are rejected cleanly. The system clock is never
    consulted and no "today" is ever inferred.

    The query matches the exact stored ISO ``transaction_date`` text:
    transactions on adjacent previous or next dates are excluded.
    Within the single date, rows are ordered newest-first by ``id
    DESC``. Only ``TransactionStatus.ACTIVE`` rows are returned.
    Returns ``()`` when no ACTIVE transaction exists on that date.

    Read-only: no rows are written and no transaction is started,
    committed, or rolled back.
    """
    return list_active_transactions_by_date(connection, transaction_date)


def list_transactions_by_month(
    connection: sqlite3.Connection,
    *,
    year: int,
    month: int,
) -> tuple[Transaction, ...]:
    """Return the ACTIVE transactions of one exact calendar month.

    ``year`` must be a real ``int`` between 1 and 9999 (the years
    representable by :class:`datetime.date`) and ``month`` a real
    ``int`` between 1 and 12; ``bool`` values, floats, strings, and
    ``None`` are rejected deterministically without coercion.

    Month boundaries are computed exactly with the stdlib calendar
    (first day through last day, leap years included); the
    December -> January rollover is exact, and ``year=9999,
    month=12`` works without ever constructing a year 10000 date.
    Transactions in the previous and next month are excluded.

    Results contain only ``TransactionStatus.ACTIVE`` rows ordered
    newest-first by ``transaction_date DESC, id DESC``. Returns
    ``()`` when no ACTIVE transaction exists in that month.

    Read-only: no rows are written and no transaction is started,
    committed, or rolled back.
    """
    return list_active_transactions_in_month(connection, year=year, month=month)
