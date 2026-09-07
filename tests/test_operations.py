"""Tests for the read-only Finance query operations (stage D1).

All tests are deterministic: no network, no wall-clock time, no
randomness, no environment access. Filesystem mutation happens only
inside pytest-owned temporary directories or in in-memory SQLite
databases (``:memory:``).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

import pytest

import hermes_finance
import hermes_finance.operations as operations_module
from hermes_finance import (
    Direction,
    FinanceConfig,
    TelegramMessageRef,
    Transaction,
    TransactionStatus,
    list_recent_transactions,
    list_transactions_by_date,
    list_transactions_by_month,
    open_database,
)
from hermes_finance.repository import (
    RepositoryDataError,
    get_transaction,
    persist_transaction,
)

OPERATIONS_SOURCE: str = Path(operations_module.__file__).read_text(encoding="utf-8")

FIXED_TZ = UTC
FIXED_CREATED_AT = datetime(2026, 9, 1, 10, 0, 0, tzinfo=FIXED_TZ)
FIXED_UPDATED_AT = datetime(2026, 9, 1, 11, 30, 0, tzinfo=FIXED_TZ)
FIXED_PROCESSED_AT = datetime(2026, 9, 1, 12, 15, 0, tzinfo=FIXED_TZ)
FIXED_DELETED_AT_TEXT = "2026-09-01T13:00:00+00:00"

LEAP_DAY: Final[date] = date(2028, 2, 29)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def conn() -> Iterator[sqlite3.Connection]:
    """A fresh migrated in-memory database connection."""
    connection = open_database(
        FinanceConfig(database_path=Path(":memory:"), business_timezone=FIXED_TZ)
    )
    try:
        yield connection
    finally:
        connection.close()


def make_ref(*, message_id: int, update_id: int) -> TelegramMessageRef:
    """A valid Telegram message provenance reference."""
    return TelegramMessageRef(
        chat_id=-100200,
        message_thread_id=7,
        message_id=message_id,
        update_id=update_id,
    )


def make_transaction(**overrides: Any) -> Transaction:
    """A valid unpersisted Transaction with field overrides."""
    values: dict[str, Any] = {
        "direction": Direction.EXPENSE,
        "amount_usdt": Decimal(25),
        "category": "Groceries",
        "source": "card",
        "comment": None,
        "transaction_date": date(2026, 9, 1),
        "created_at": FIXED_CREATED_AT,
        "updated_at": FIXED_UPDATED_AT,
        "status": TransactionStatus.ACTIVE,
        "deleted_at": None,
    }
    values.update(overrides)
    return Transaction(**values)


def persist_row(
    connection: sqlite3.Connection,
    *,
    message_id: int,
    transaction_date: date,
    **transaction_overrides: Any,
) -> Transaction:
    """Persist one ACTIVE transaction with the given date and overrides."""
    return persist_transaction(
        connection,
        make_transaction(transaction_date=transaction_date, **transaction_overrides),
        make_ref(message_id=message_id, update_id=10_000 + message_id),
        processed_at=FIXED_PROCESSED_AT,
    )


def insert_raw_row(
    connection: sqlite3.Connection,
    *,
    message_id: int,
    transaction_date: str,
    status: str = "active",
    amount_usdt: Any = "25",
    update_id: int | None = None,
) -> None:
    """Insert one raw transactions row, bypassing the domain layer.

    ``status='deleted'`` rows get a deterministic ``deleted_at`` so the
    schema-v1 CHECK constraint holds. Precise-ordering, DELETED-row and
    corrupt-row fixtures rely on this direct SQL path.
    """
    deleted_at: str | None = None
    if status == "deleted":
        deleted_at = FIXED_DELETED_AT_TEXT
    connection.execute(
        "INSERT INTO transactions ("
        " direction, amount_usdt, category, source, comment,"
        " transaction_date, created_at, updated_at, status, deleted_at,"
        " chat_id, message_thread_id, message_id, update_id"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "expense",
            amount_usdt,
            "Groceries",
            "card",
            None,
            transaction_date,
            "2026-09-01T10:00:00+00:00",
            "2026-09-01T10:00:00+00:00",
            status,
            deleted_at,
            -100200,
            7,
            message_id,
            20_000 + message_id if update_id is None else update_id,
        ),
    )
    connection.commit()


def insert_many_raw_rows(
    connection: sqlite3.Connection, *, count: int, transaction_date: str
) -> None:
    """Insert ``count`` raw ACTIVE transactions rows on one date."""
    connection.executemany(
        "INSERT INTO transactions ("
        " direction, amount_usdt, category, source, comment,"
        " transaction_date, created_at, updated_at, status, deleted_at,"
        " chat_id, message_thread_id, message_id, update_id"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "expense",
                "25",
                "Groceries",
                "card",
                None,
                transaction_date,
                "2026-09-01T10:00:00+00:00",
                "2026-09-01T10:00:00+00:00",
                "active",
                None,
                -100200,
                7,
                1_000 + index,
                30_000 + index,
            )
            for index in range(count)
        ],
    )
    connection.commit()


def count_transactions(connection: sqlite3.Connection) -> int:
    """Number of transactions rows."""
    row = connection.execute("SELECT COUNT(*) FROM transactions").fetchone()
    return int(row[0])


def ids_of(result: Sequence[Transaction]) -> list[str]:
    """The transaction_id sequence of a query result, in order."""
    return [transaction.transaction_id or "" for transaction in result]


ALL_OPERATIONS: Final[list[Callable[[sqlite3.Connection], tuple[Transaction, ...]]]] = [
    lambda connection: list_recent_transactions(connection),
    lambda connection: list_recent_transactions(connection, limit=3),
    lambda connection: list_transactions_by_date(connection, date(2026, 9, 1)),
    lambda connection: list_transactions_by_month(connection, year=2026, month=9),
]


# ---------------------------------------------------------------------------
# connection contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operation", ALL_OPERATIONS)
@pytest.mark.parametrize("bad_connection", ["not-a-connection", None, 5, object()])
def test_operations_reject_non_connection(
    operation: Callable[[sqlite3.Connection], tuple[Transaction, ...]],
    bad_connection: Any,
) -> None:
    with pytest.raises(TypeError):
        operation(bad_connection)


class _FakeConnection:
    """An object that mimics part of the connection interface."""

    in_transaction = False


@pytest.mark.parametrize("operation", ALL_OPERATIONS)
def test_operations_reject_lookalike_connection_object(
    operation: Callable[[sqlite3.Connection], tuple[Transaction, ...]],
) -> None:
    fake: Any = _FakeConnection()
    with pytest.raises(TypeError):
        operation(fake)


# ---------------------------------------------------------------------------
# recent transactions ("last")
# ---------------------------------------------------------------------------


def test_recent_empty_database_returns_empty_tuple(conn: sqlite3.Connection) -> None:
    assert list_recent_transactions(conn) == ()


def test_recent_returns_all_when_fewer_than_limit(conn: sqlite3.Connection) -> None:
    persist_row(conn, message_id=1, transaction_date=date(2026, 9, 1))
    persist_row(conn, message_id=2, transaction_date=date(2026, 9, 2))
    persist_row(conn, message_id=3, transaction_date=date(2026, 9, 3))
    result = list_recent_transactions(conn)
    assert len(result) == 3


def test_recent_default_limit_is_ten(conn: sqlite3.Connection) -> None:
    for index in range(12):
        persist_row(conn, message_id=index + 1, transaction_date=date(2026, 9, 1))
    result = list_recent_transactions(conn)
    assert len(result) == 10
    # Newest-first within the single date: SQLite ids 12 down to 3.
    assert ids_of(result) == [str(value) for value in range(12, 2, -1)]


def test_recent_custom_valid_limit(conn: sqlite3.Connection) -> None:
    for index in range(7):
        persist_row(conn, message_id=index + 1, transaction_date=date(2026, 9, 1))
    result = list_recent_transactions(conn, limit=5)
    assert len(result) == 5
    assert ids_of(result) == ["7", "6", "5", "4", "3"]


def test_recent_max_limit_100_accepted(conn: sqlite3.Connection) -> None:
    insert_many_raw_rows(conn, count=101, transaction_date="2026-09-01")
    result = list_recent_transactions(conn, limit=100)
    assert len(result) == 100
    # The 100 newest of the 101 rows: ids 101 down to 2.
    assert ids_of(result)[:3] == ["101", "100", "99"]
    assert ids_of(result)[-1] == "2"


@pytest.mark.parametrize("bad_limit", [0, -1, -100, 101, 1000])
def test_recent_rejects_out_of_range_limit(
    conn: sqlite3.Connection, bad_limit: int
) -> None:
    with pytest.raises(ValueError):
        list_recent_transactions(conn, limit=bad_limit)


@pytest.mark.parametrize("bad_limit", [True, False])
def test_recent_rejects_bool_limit(conn: sqlite3.Connection, bad_limit: bool) -> None:
    with pytest.raises(TypeError):
        list_recent_transactions(conn, limit=bad_limit)


@pytest.mark.parametrize("bad_limit", [1.5, 10.0, "10", None, [10]])
def test_recent_rejects_non_int_limit(conn: sqlite3.Connection, bad_limit: Any) -> None:
    with pytest.raises(TypeError):
        list_recent_transactions(conn, limit=bad_limit)


def test_recent_rejects_invalid_limit_before_any_sql(conn: sqlite3.Connection) -> None:
    # Rejection happens deterministically before SQL execution: a broken
    # connection never receives a query when the limit is invalid.
    with pytest.raises((TypeError, ValueError)):
        list_recent_transactions("not-a-connection", limit=0)  # type: ignore[arg-type]


def test_recent_deterministic_ordering_date_desc_then_id_desc(
    conn: sqlite3.Connection,
) -> None:
    persist_row(conn, message_id=1, transaction_date=date(2026, 9, 1))
    persist_row(conn, message_id=2, transaction_date=date(2026, 9, 1))
    persist_row(conn, message_id=3, transaction_date=date(2026, 9, 2))
    persist_row(conn, message_id=4, transaction_date=date(2026, 8, 31))
    persist_row(conn, message_id=5, transaction_date=date(2026, 9, 1))
    result = list_recent_transactions(conn)
    # Newest business date first; within one date newest SQLite id first.
    assert ids_of(result) == ["3", "5", "2", "1", "4"]


def test_recent_preserves_domain_fields_exactly(conn: sqlite3.Connection) -> None:
    persisted = persist_row(
        conn,
        message_id=1,
        transaction_date=date(2026, 9, 1),
        direction=Direction.INCOME,
        amount_usdt=Decimal("123.456789"),
        category="Salary",
        source="bank",
        comment="note",
    )
    result = list_recent_transactions(conn)
    assert len(result) == 1
    assert result[0] == persisted
    assert result[0].amount_usdt == Decimal("123.456789")
    assert result[0].status is TransactionStatus.ACTIVE
    assert result[0].comment == "note"


# ---------------------------------------------------------------------------
# ACTIVE-only visibility
# ---------------------------------------------------------------------------


def _persist_active_and_deleted_rows(conn: sqlite3.Connection) -> None:
    """Three ACTIVE rows plus two newer DELETED rows (ids 4 and 5)."""
    persist_row(conn, message_id=1, transaction_date=date(2026, 9, 1))
    persist_row(conn, message_id=2, transaction_date=date(2026, 9, 10))
    persist_row(conn, message_id=3, transaction_date=date(2026, 9, 20))
    insert_raw_row(
        conn, message_id=4, transaction_date="2026-09-25", status="deleted"
    )
    insert_raw_row(
        conn, message_id=5, transaction_date="2026-09-30", status="deleted"
    )


def test_recent_excludes_deleted_rows(conn: sqlite3.Connection) -> None:
    _persist_active_and_deleted_rows(conn)
    result = list_recent_transactions(conn)
    # The DELETED rows are the two newest; they must be invisible, and
    # the remaining order is still newest-first.
    assert ids_of(result) == ["3", "2", "1"]


def test_by_date_excludes_deleted_rows(conn: sqlite3.Connection) -> None:
    persist_row(conn, message_id=1, transaction_date=date(2026, 9, 15))
    insert_raw_row(
        conn, message_id=2, transaction_date="2026-09-15", status="deleted"
    )
    result = list_transactions_by_date(conn, date(2026, 9, 15))
    assert ids_of(result) == ["1"]


def test_by_month_excludes_deleted_rows(conn: sqlite3.Connection) -> None:
    _persist_active_and_deleted_rows(conn)
    result = list_transactions_by_month(conn, year=2026, month=9)
    assert ids_of(result) == ["3", "2", "1"]


def test_low_level_get_transaction_still_returns_deleted_row(
    conn: sqlite3.Connection,
) -> None:
    insert_raw_row(
        conn, message_id=1, transaction_date="2026-09-15", status="deleted"
    )
    deleted = get_transaction(conn, "1")
    assert deleted is not None
    assert deleted.status is TransactionStatus.DELETED
    assert deleted.deleted_at is not None
    # D1 normal queries do not see it.
    assert list_transactions_by_date(conn, date(2026, 9, 15)) == ()


# ---------------------------------------------------------------------------
# exact date query
# ---------------------------------------------------------------------------


def test_by_date_exact_match_with_leap_day_and_adjacent_dates_excluded(
    conn: sqlite3.Connection,
) -> None:
    persist_row(conn, message_id=1, transaction_date=date(2028, 2, 28))
    persist_row(conn, message_id=2, transaction_date=LEAP_DAY)
    persist_row(conn, message_id=3, transaction_date=LEAP_DAY)
    persist_row(conn, message_id=4, transaction_date=date(2028, 3, 1))
    result = list_transactions_by_date(conn, LEAP_DAY)
    # Exact match only; deterministic id DESC within the date.
    assert ids_of(result) == ["3", "2"]


def test_by_date_empty_result_returns_empty_tuple(conn: sqlite3.Connection) -> None:
    persist_row(conn, message_id=1, transaction_date=date(2026, 9, 1))
    assert list_transactions_by_date(conn, date(2026, 9, 2)) == ()


def test_by_date_empty_database_returns_empty_tuple(conn: sqlite3.Connection) -> None:
    assert list_transactions_by_date(conn, LEAP_DAY) == ()


def test_by_date_rejects_datetime(conn: sqlite3.Connection) -> None:
    with pytest.raises(TypeError):
        list_transactions_by_date(
            conn,
            datetime(2028, 2, 29, 10, 0, 0, tzinfo=timezone(timedelta(hours=2))),
        )


@pytest.mark.parametrize("bad_date", ["2028-02-29", None, 5, object()])
def test_by_date_rejects_non_date_values(
    conn: sqlite3.Connection, bad_date: Any
) -> None:
    with pytest.raises(TypeError):
        list_transactions_by_date(conn, bad_date)


# ---------------------------------------------------------------------------
# calendar month query
# ---------------------------------------------------------------------------


def test_month_normal_boundaries_first_and_last_day_included(
    conn: sqlite3.Connection,
) -> None:
    persist_row(conn, message_id=1, transaction_date=date(2026, 8, 31))
    persist_row(conn, message_id=2, transaction_date=date(2026, 9, 1))
    persist_row(conn, message_id=3, transaction_date=date(2026, 9, 30))
    persist_row(conn, message_id=4, transaction_date=date(2026, 10, 1))
    result = list_transactions_by_month(conn, year=2026, month=9)
    # Previous and next month excluded; newest-first inside the month.
    assert ids_of(result) == ["3", "2"]


def test_month_february_non_leap_boundaries(conn: sqlite3.Connection) -> None:
    persist_row(conn, message_id=1, transaction_date=date(2027, 1, 31))
    persist_row(conn, message_id=2, transaction_date=date(2027, 2, 1))
    persist_row(conn, message_id=3, transaction_date=date(2027, 2, 28))
    persist_row(conn, message_id=4, transaction_date=date(2027, 3, 1))
    result = list_transactions_by_month(conn, year=2027, month=2)
    assert ids_of(result) == ["3", "2"]


def test_month_february_leap_year_includes_29th(conn: sqlite3.Connection) -> None:
    persist_row(conn, message_id=1, transaction_date=date(2028, 2, 28))
    persist_row(conn, message_id=2, transaction_date=date(2028, 2, 29))
    persist_row(conn, message_id=3, transaction_date=date(2028, 3, 1))
    result = list_transactions_by_month(conn, year=2028, month=2)
    assert ids_of(result) == ["2", "1"]


def test_month_december_to_january_rollover(conn: sqlite3.Connection) -> None:
    persist_row(conn, message_id=1, transaction_date=date(2026, 11, 30))
    persist_row(conn, message_id=2, transaction_date=date(2026, 12, 1))
    persist_row(conn, message_id=3, transaction_date=date(2026, 12, 31))
    persist_row(conn, message_id=4, transaction_date=date(2027, 1, 1))
    december = list_transactions_by_month(conn, year=2026, month=12)
    assert ids_of(december) == ["3", "2"]
    january = list_transactions_by_month(conn, year=2027, month=1)
    assert ids_of(january) == ["4"]


def test_month_year_9999_december_includes_last_day_without_year_10000(
    conn: sqlite3.Connection,
) -> None:
    insert_raw_row(conn, message_id=1, transaction_date="9999-11-30")
    insert_raw_row(conn, message_id=2, transaction_date="9999-12-31")
    # A text date beyond year 9999 cannot be produced by the domain
    # layer; direct SQL proves the query boundary stays bounded.
    insert_raw_row(conn, message_id=3, transaction_date="10000-01-01")
    result = list_transactions_by_month(conn, year=9999, month=12)
    assert ids_of(result) == ["2"]


def test_month_empty_database_returns_empty_tuple(conn: sqlite3.Connection) -> None:
    assert list_transactions_by_month(conn, year=2026, month=9) == ()


@pytest.mark.parametrize("bad_month", [0, 13, -1, 100])
def test_month_rejects_out_of_range_month(
    conn: sqlite3.Connection, bad_month: int
) -> None:
    with pytest.raises(ValueError):
        list_transactions_by_month(conn, year=2026, month=bad_month)


@pytest.mark.parametrize("bad_month", [True, False, 1.5, "9", None])
def test_month_rejects_invalid_month_types(
    conn: sqlite3.Connection, bad_month: Any
) -> None:
    with pytest.raises(TypeError):
        list_transactions_by_month(conn, year=2026, month=bad_month)


@pytest.mark.parametrize("bad_year", [0, -1, 10000, 100000])
def test_month_rejects_out_of_range_year(
    conn: sqlite3.Connection, bad_year: int
) -> None:
    with pytest.raises(ValueError):
        list_transactions_by_month(conn, year=bad_year, month=6)


@pytest.mark.parametrize("bad_year", [True, False, 2026.0, "2026", None])
def test_month_rejects_invalid_year_types(
    conn: sqlite3.Connection, bad_year: Any
) -> None:
    with pytest.raises(TypeError):
        list_transactions_by_month(conn, year=bad_year, month=6)


def test_month_accepts_representable_year_bounds(conn: sqlite3.Connection) -> None:
    assert list_transactions_by_month(conn, year=1, month=1) == ()
    assert list_transactions_by_month(conn, year=9999, month=12) == ()


# ---------------------------------------------------------------------------
# side effects and read-only contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operation", ALL_OPERATIONS)
def test_operations_do_not_change_database_content_or_transaction_state(
    operation: Callable[[sqlite3.Connection], tuple[Transaction, ...]],
    conn: sqlite3.Connection,
) -> None:
    persist_row(conn, message_id=1, transaction_date=date(2026, 9, 1))
    persist_row(conn, message_id=2, transaction_date=date(2026, 9, 15))
    before_transactions = conn.execute(
        "SELECT * FROM transactions ORDER BY id"
    ).fetchall()
    before_updates = conn.execute(
        "SELECT * FROM processed_updates ORDER BY update_id"
    ).fetchall()
    assert conn.in_transaction is False

    operation(conn)

    assert conn.in_transaction is False
    assert (
        conn.execute("SELECT * FROM transactions ORDER BY id").fetchall()
        == before_transactions
    )
    assert conn.execute(
        "SELECT * FROM processed_updates ORDER BY update_id"
    ).fetchall() == before_updates
    assert count_transactions(conn) == 2


CALLER_TRANSACTION_PROBES: Final[
    list[tuple[Callable[[sqlite3.Connection], tuple[Transaction, ...]], int, str]]
] = [
    (lambda connection: list_recent_transactions(connection), 4, "4"),
    (
        lambda connection: list_transactions_by_date(connection, date(2026, 9, 15)),
        1,
        "4",
    ),
    (lambda connection: list_transactions_by_month(connection, year=2026, month=9), 4, "4"),
]


@pytest.mark.parametrize(
    ("operation", "expected_count", "pending_id"),
    CALLER_TRANSACTION_PROBES,
)
def test_operations_leave_caller_active_transaction_untouched(
    operation: Callable[[sqlite3.Connection], tuple[Transaction, ...]],
    expected_count: int,
    pending_id: str,
    conn: sqlite3.Connection,
) -> None:
    persist_row(conn, message_id=1, transaction_date=date(2026, 9, 10))
    persist_row(conn, message_id=2, transaction_date=date(2026, 9, 11))
    persist_row(conn, message_id=3, transaction_date=date(2026, 9, 12))

    # Caller begins a transaction and writes one pending row.
    conn.execute("BEGIN")
    insert_pending_row(conn, message_id=9, transaction_date="2026-09-15")
    assert conn.in_transaction is True

    # The D1 query reads within the caller transaction: the pending row
    # is visible on the same connection, and the caller transaction is
    # still active afterwards.
    result = operation(conn)
    assert conn.in_transaction is True
    assert len(result) == expected_count
    assert pending_id in ids_of(result)

    # The caller can still roll back its own work afterwards.
    conn.rollback()
    assert conn.in_transaction is False
    assert count_transactions(conn) == 3
    assert get_transaction(conn, "4") is None


def insert_pending_row(
    connection: sqlite3.Connection, *, message_id: int, transaction_date: str
) -> None:
    """Insert one raw pending transactions row without committing."""
    connection.execute(
        "INSERT INTO transactions ("
        " direction, amount_usdt, category, source, comment,"
        " transaction_date, created_at, updated_at, status, deleted_at,"
        " chat_id, message_thread_id, message_id, update_id"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "expense",
            "25",
            "Groceries",
            "card",
            None,
            transaction_date,
            "2026-09-01T10:00:00+00:00",
            "2026-09-01T10:00:00+00:00",
            "active",
            None,
            -100200,
            7,
            message_id,
            40_000 + message_id,
        ),
    )


# ---------------------------------------------------------------------------
# corrupt stored rows
# ---------------------------------------------------------------------------


def test_corrupt_row_raises_repository_data_error_with_cause(
    conn: sqlite3.Connection,
) -> None:
    persist_row(conn, message_id=1, transaction_date=date(2026, 9, 1))
    insert_raw_row(conn, message_id=2, transaction_date="2026-09-02", amount_usdt="abc")
    with pytest.raises(RepositoryDataError) as excinfo:
        list_recent_transactions(conn)
    assert excinfo.value.__cause__ is not None


def test_corrupt_row_by_month_raises_repository_data_error(
    conn: sqlite3.Connection,
) -> None:
    insert_raw_row(conn, message_id=1, transaction_date="2026-09-02", amount_usdt="abc")
    with pytest.raises(RepositoryDataError) as excinfo:
        list_transactions_by_month(conn, year=2026, month=9)
    assert excinfo.value.__cause__ is not None


def test_corrupt_row_outside_query_range_does_not_raise(
    conn: sqlite3.Connection,
) -> None:
    persist_row(conn, message_id=1, transaction_date=date(2026, 9, 1))
    # A corrupt row in another month is never selected, so it cannot
    # and must not turn the September result into an error.
    insert_raw_row(conn, message_id=2, transaction_date="2025-01-02", amount_usdt="abc")
    result = list_transactions_by_month(conn, year=2026, month=9)
    assert ids_of(result) == ["1"]


# ---------------------------------------------------------------------------
# public API surface
# ---------------------------------------------------------------------------


def test_package_exports_operations_deliberately() -> None:
    for name in (
        "list_recent_transactions",
        "list_transactions_by_date",
        "list_transactions_by_month",
    ):
        assert name in hermes_finance.__all__
        assert callable(getattr(hermes_finance, name))


def test_repository_all_contract_unchanged() -> None:
    import hermes_finance.repository as repository_module

    assert repository_module.__all__ == [
        "RepositoryDataError",
        "RepositoryTransactionError",
        "find_transaction_by_message",
        "find_transaction_by_update",
        "get_transaction",
        "is_update_processed",
        "persist_transaction",
    ]


@pytest.mark.parametrize(
    "forbidden",
    [
        "os.environ",
        "getenv",
        "datetime.now",
        "datetime.utcnow",
        "datetime.today",
        "date.today",
        "today()",
        "time.time",
        "time.monotonic",
        "perf_counter",
        "monotonic",
        "import os",
        "import time",
        "sqlite3.connect",
        "PRAGMA",
        "BEGIN",
        "COMMIT",
        "ROLLBACK",
        "INSERT",
        "UPDATE ",
        "DELETE",
    ],
)
def test_operations_source_avoids_forbidden_operations(forbidden: str) -> None:
    assert forbidden not in OPERATIONS_SOURCE


def test_operations_public_api_is_small_and_deliberate() -> None:
    assert operations_module.__all__ == [
        "list_recent_transactions",
        "list_transactions_by_date",
        "list_transactions_by_month",
    ]
