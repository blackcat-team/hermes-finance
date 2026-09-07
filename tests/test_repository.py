"""Tests for the SQLite repository layer (stage C2).

All tests are deterministic: no network, no wall-clock time, no
randomness, no environment access. Filesystem mutation happens only
inside pytest-owned temporary directories (``tmp_path``) or in in-memory
SQLite databases (``:memory:``).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

import pytest

import hermes_finance.repository as repository_module
from hermes_finance import (
    Direction,
    FinanceConfig,
    TelegramMessageIdentity,
    TelegramMessageRef,
    TelegramUpdateIdentity,
    Transaction,
    TransactionStatus,
    open_database,
)
from hermes_finance.repository import (
    RepositoryDataError,
    RepositoryTransactionError,
    find_transaction_by_message,
    find_transaction_by_update,
    get_transaction,
    is_update_processed,
    persist_transaction,
)

REPOSITORY_SOURCE: str = Path(repository_module.__file__).read_text(encoding="utf-8")

FIXED_TZ = UTC
NON_UTC_TZ = timezone(timedelta(hours=5, minutes=30))

FIXED_DATE = date(2026, 9, 1)
FIXED_CREATED_AT = datetime(2026, 9, 1, 10, 0, 0, tzinfo=FIXED_TZ)
FIXED_UPDATED_AT = datetime(2026, 9, 1, 11, 30, 0, tzinfo=FIXED_TZ)
FIXED_PROCESSED_AT = datetime(2026, 9, 1, 12, 15, 0, tzinfo=FIXED_TZ)
FIXED_DELETED_AT = datetime(2026, 9, 1, 13, 45, 0, tzinfo=NON_UTC_TZ)

DECIMAL_CASES: Final[list[str]] = [
    "25",
    "0.000001",
    "123.456789",
    "123.4500",
    "1.234567890123456789012345789",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def memory_config() -> FinanceConfig:
    """A valid config pointing at a fresh in-memory database."""
    return FinanceConfig(database_path=Path(":memory:"), business_timezone=FIXED_TZ)


def open_memory() -> sqlite3.Connection:
    """Open a fresh migrated in-memory database connection."""
    return open_database(memory_config())


def file_config(path: Path) -> FinanceConfig:
    """A valid config pointing at a file-backed database path."""
    return FinanceConfig(database_path=path, business_timezone=FIXED_TZ)


def make_ref(
    *,
    chat_id: int = -100200,
    message_thread_id: int = 7,
    message_id: int = 1,
    update_id: int = 1000,
) -> TelegramMessageRef:
    """A valid Telegram message provenance reference."""
    return TelegramMessageRef(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
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
        "transaction_date": FIXED_DATE,
        "created_at": FIXED_CREATED_AT,
        "updated_at": FIXED_UPDATED_AT,
        "status": TransactionStatus.ACTIVE,
        "deleted_at": None,
    }
    values.update(overrides)
    return Transaction(**values)


def persist_fixed(
    conn: sqlite3.Connection,
    *,
    chat_id: int = -100200,
    message_thread_id: int = 7,
    message_id: int = 1,
    update_id: int = 1000,
    transaction: Transaction | None = None,
    processed_at: datetime | None = None,
) -> Transaction:
    """persist_transaction with the fixed test values and overrides."""
    return persist_transaction(
        conn,
        transaction if transaction is not None else make_transaction(),
        make_ref(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            message_id=message_id,
            update_id=update_id,
        ),
        processed_at=processed_at if processed_at is not None else FIXED_PROCESSED_AT,
    )


def count_transactions(conn: sqlite3.Connection) -> int:
    """Number of transactions rows."""
    row = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()
    return int(row[0])


def count_processed_updates(conn: sqlite3.Connection) -> int:
    """Number of processed_updates rows."""
    row = conn.execute("SELECT COUNT(*) FROM processed_updates").fetchone()
    return int(row[0])


def transaction_row(conn: sqlite3.Connection, message_id: int) -> Any:
    """The full transactions row for the given message_id, or None."""
    return conn.execute(
        "SELECT * FROM transactions WHERE message_id = ?", (message_id,)
    ).fetchone()


def processed_update_row(conn: sqlite3.Connection, update_id: int) -> Any:
    """The full processed_updates row for the given update_id, or None."""
    return conn.execute(
        "SELECT * FROM processed_updates WHERE update_id = ?", (update_id,)
    ).fetchone()


def insert_raw_transaction(conn: sqlite3.Connection, **overrides: Any) -> None:
    """Insert one raw transactions row (bypassing the domain layer)."""
    values: dict[str, Any] = {
        "direction": "expense",
        "amount_usdt": "25",
        "category": "Groceries",
        "source": "card",
        "comment": None,
        "transaction_date": "2026-09-01",
        "created_at": "2026-09-01T10:00:00+00:00",
        "updated_at": "2026-09-01T10:00:00+00:00",
        "status": "active",
        "deleted_at": None,
        "chat_id": -100200,
        "message_thread_id": 7,
        "message_id": 900,
        "update_id": 9000,
    }
    values.update(overrides)
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    conn.execute(
        f"INSERT INTO transactions ({columns}) VALUES ({placeholders})",
        tuple(values.values()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# connection contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_connection", ["not-a-connection", None, 5, object()])
def test_persist_transaction_rejects_non_connection(bad_connection: Any) -> None:
    with pytest.raises(TypeError):
        persist_transaction(
            bad_connection, make_transaction(), make_ref(), processed_at=FIXED_PROCESSED_AT
        )


READ_CALLS: Final[list[Callable[[Any], object]]] = [
    lambda conn: get_transaction(conn, "1"),
    lambda conn: find_transaction_by_message(
        conn, TelegramMessageIdentity(chat_id=-100200, message_id=1)
    ),
    lambda conn: find_transaction_by_update(conn, TelegramUpdateIdentity(update_id=1000)),
    lambda conn: is_update_processed(conn, TelegramUpdateIdentity(update_id=1000)),
]


@pytest.mark.parametrize("read_call", READ_CALLS)
@pytest.mark.parametrize("bad_connection", ["not-a-connection", None, 5, object()])
def test_read_functions_reject_non_connection(
    read_call: Callable[[Any], object], bad_connection: Any
) -> None:
    with pytest.raises(TypeError):
        read_call(bad_connection)


class _FakeConnection:
    """An object that mimics part of the connection interface."""

    in_transaction = False


def test_persist_transaction_rejects_lookalike_connection_object() -> None:
    fake: Any = _FakeConnection()
    with pytest.raises(TypeError):
        persist_transaction(
            fake, make_transaction(), make_ref(), processed_at=FIXED_PROCESSED_AT
        )


# ---------------------------------------------------------------------------
# persist input contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_transaction", [None, "transaction", 5, object()])
def test_persist_transaction_rejects_non_transaction(bad_transaction: Any) -> None:
    conn = open_memory()
    try:
        with pytest.raises(TypeError):
            persist_transaction(
                conn, bad_transaction, make_ref(), processed_at=FIXED_PROCESSED_AT
            )
    finally:
        conn.close()


@pytest.mark.parametrize("bad_provenance", [None, "ref", 5, object()])
def test_persist_transaction_rejects_non_provenance(bad_provenance: Any) -> None:
    conn = open_memory()
    try:
        with pytest.raises(TypeError):
            persist_transaction(
                conn, make_transaction(), bad_provenance, processed_at=FIXED_PROCESSED_AT
            )
    finally:
        conn.close()


def test_persist_transaction_rejects_already_persisted_transaction() -> None:
    conn = open_memory()
    try:
        persisted = persist_fixed(conn, message_id=1, update_id=1000)
        with pytest.raises(RepositoryDataError):
            persist_transaction(
                conn,
                persisted,
                make_ref(message_id=2, update_id=1001),
                processed_at=FIXED_PROCESSED_AT,
            )
        assert count_transactions(conn) == 1
        assert count_processed_updates(conn) == 1
    finally:
        conn.close()


def test_persist_transaction_rejects_naive_processed_at() -> None:
    conn = open_memory()
    try:
        naive = datetime(2026, 9, 1, 12, 15, 0)  # noqa: DTZ001 - naive is the point
        with pytest.raises(ValueError):
            persist_transaction(conn, make_transaction(), make_ref(), processed_at=naive)
        assert count_transactions(conn) == 0
        assert count_processed_updates(conn) == 0
    finally:
        conn.close()


@pytest.mark.parametrize(
    "bad_processed_at", [None, "2026-09-01T12:15:00+00:00", 1782000000]
)
def test_persist_transaction_rejects_non_datetime_processed_at(bad_processed_at: Any) -> None:
    conn = open_memory()
    try:
        with pytest.raises(TypeError):
            persist_transaction(
                conn, make_transaction(), make_ref(), processed_at=bad_processed_at
            )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# atomic write: success
# ---------------------------------------------------------------------------


def test_persist_active_transaction_creates_both_rows() -> None:
    conn = open_memory()
    try:
        result = persist_fixed(conn, message_id=1, update_id=1000)
        assert result.status is TransactionStatus.ACTIVE
        assert count_transactions(conn) == 1
        assert count_processed_updates(conn) == 1
        assert conn.in_transaction is False
    finally:
        conn.close()


def test_persist_transaction_row_fields_are_correct() -> None:
    conn = open_memory()
    try:
        persist_fixed(conn, message_id=5, update_id=1005)
        row = transaction_row(conn, message_id=5)
        assert row is not None
        assert row[1] == "expense"  # direction
        assert row[2] == "25"  # amount_usdt
        assert row[3] == "Groceries"  # category
        assert row[4] == "card"  # source
        assert row[5] is None  # comment
        assert row[6] == "2026-09-01"  # transaction_date
        assert row[7] == "2026-09-01T10:00:00+00:00"  # created_at
        assert row[8] == "2026-09-01T11:30:00+00:00"  # updated_at
        assert row[9] == "active"  # status
        assert row[10] is None  # deleted_at
        assert row[11] == -100200  # chat_id
        assert row[12] == 7  # message_thread_id
        assert row[13] == 5  # message_id
        assert row[14] == 1005  # update_id
    finally:
        conn.close()


def test_persist_processed_updates_row_fields_are_correct() -> None:
    conn = open_memory()
    try:
        persist_fixed(conn, message_id=5, update_id=1005)
        row = processed_update_row(conn, update_id=1005)
        assert row is not None
        assert row[0] == 1005  # update_id
        assert row[1] == -100200  # chat_id
        assert row[2] == 7  # message_thread_id
        assert row[3] == 5  # message_id
        assert row[4] == "2026-09-01T12:15:00+00:00"  # processed_at
    finally:
        conn.close()


def test_persist_returns_transaction_with_string_row_id() -> None:
    conn = open_memory()
    try:
        result = persist_fixed(conn, message_id=1, update_id=1000)
        row = transaction_row(conn, message_id=1)
        assert row is not None
        assert isinstance(result.transaction_id, str)
        assert result.transaction_id == str(row[0])
        assert result == make_transaction(transaction_id=result.transaction_id)
    finally:
        conn.close()


def test_persist_original_input_remains_unpersisted() -> None:
    conn = open_memory()
    try:
        original = make_transaction()
        persist_transaction(conn, original, make_ref(), processed_at=FIXED_PROCESSED_AT)
        assert original.transaction_id is None
        assert original == make_transaction()
    finally:
        conn.close()


def test_persist_returns_new_object() -> None:
    conn = open_memory()
    try:
        original = make_transaction()
        result = persist_transaction(conn, original, make_ref(), processed_at=FIXED_PROCESSED_AT)
        assert result is not original
    finally:
        conn.close()


def test_persist_commit_survives_close_and_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "finance.sqlite"
    conn = open_database(file_config(db_path))
    try:
        result = persist_fixed(conn, message_id=1, update_id=1000)
    finally:
        conn.close()

    reopened = open_database(file_config(db_path))
    try:
        assert count_transactions(reopened) == 1
        assert count_processed_updates(reopened) == 1
        loaded = get_transaction(reopened, "1")
        assert loaded == result
        assert is_update_processed(reopened, TelegramUpdateIdentity(update_id=1000)) is True
    finally:
        reopened.close()


def test_persist_deleted_valid_transaction() -> None:
    conn = open_memory()
    try:
        deleted = make_transaction(status=TransactionStatus.DELETED, deleted_at=FIXED_DELETED_AT)
        result = persist_fixed(conn, message_id=6, update_id=1006, transaction=deleted)
        assert result.status is TransactionStatus.DELETED
        assert result.deleted_at == FIXED_DELETED_AT
        assert count_transactions(conn) == 1
        row = transaction_row(conn, message_id=6)
        assert row is not None
        assert row[9] == "deleted"
        assert row[10] == FIXED_DELETED_AT.isoformat()
    finally:
        conn.close()


def test_sequential_persists_assign_distinct_string_ids() -> None:
    conn = open_memory()
    try:
        first = persist_fixed(conn, message_id=1, update_id=1000)
        second = persist_fixed(conn, message_id=2, update_id=1001)
        assert first.transaction_id == "1"
        assert second.transaction_id == "2"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# atomic write: rollback on constraint failure
# ---------------------------------------------------------------------------


def test_duplicate_logical_message_rolls_back_processed_update() -> None:
    conn = open_memory()
    try:
        persist_fixed(conn, message_id=5, update_id=1005)
        with pytest.raises(sqlite3.IntegrityError):
            persist_fixed(conn, message_id=5, update_id=1006)
        assert conn.in_transaction is False
        assert count_transactions(conn) == 1
        assert count_processed_updates(conn) == 1
        assert processed_update_row(conn, update_id=1006) is None
    finally:
        conn.close()


def test_duplicate_update_rolls_back_transaction() -> None:
    conn = open_memory()
    try:
        persist_fixed(conn, message_id=5, update_id=1005)
        with pytest.raises(sqlite3.IntegrityError):
            persist_fixed(conn, message_id=6, update_id=1005)
        assert conn.in_transaction is False
        assert count_transactions(conn) == 1
        assert transaction_row(conn, message_id=6) is None
        assert count_processed_updates(conn) == 1
    finally:
        conn.close()


def test_processed_update_failure_rolls_back_transaction_insert() -> None:
    conn = open_memory()
    try:
        # A pre-existing processed_updates row for update 1007 makes the
        # processed_updates INSERT inside persist_transaction fail after
        # the transactions INSERT has already succeeded.
        conn.execute(
            "INSERT INTO processed_updates ("
            " update_id, chat_id, message_thread_id, message_id, processed_at"
            ") VALUES (?, ?, ?, ?, ?)",
            (1007, -100200, 7, 999, "2026-09-01T09:00:00+00:00"),
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            persist_fixed(conn, message_id=7, update_id=1007)
        assert conn.in_transaction is False
        assert count_transactions(conn) == 0
        assert transaction_row(conn, message_id=7) is None
        # The pre-existing row is untouched.
        assert count_processed_updates(conn) == 1
    finally:
        conn.close()


def test_connection_remains_usable_after_rollback() -> None:
    conn = open_memory()
    try:
        persist_fixed(conn, message_id=5, update_id=1005)
        with pytest.raises(sqlite3.IntegrityError):
            persist_fixed(conn, message_id=5, update_id=1006)
        result = persist_fixed(conn, message_id=6, update_id=1007)
        assert result.transaction_id == "2"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# atomic write: commit failure triggers rollback
# ---------------------------------------------------------------------------


class _CommitFailsConnection(sqlite3.Connection):
    """A real sqlite3.Connection whose commit() fails deterministically.

    rollback() keeps its normal behaviour, so this subclass proves that
    a commit failure inside the repository write transaction triggers
    the repository's rollback path and leaves no pending repository
    transaction. Test-only; the production code has no knowledge of it.
    """

    def commit(self) -> None:
        raise sqlite3.OperationalError("deterministic commit failure")


def test_commit_failure_inside_persist_triggers_rollback(tmp_path: Path) -> None:
    db_path = tmp_path / "finance.sqlite"

    # Create the schema v1 database first with an ordinary connection.
    bootstrap = open_database(file_config(db_path))
    try:
        assert int(bootstrap.execute("PRAGMA user_version").fetchone()[0]) == 1
    finally:
        bootstrap.close()

    conn = sqlite3.connect(db_path, factory=_CommitFailsConnection)
    try:
        # The repository receives a real sqlite3.Connection (subclass).
        assert isinstance(conn, sqlite3.Connection)
        assert type(conn) is _CommitFailsConnection
        assert conn.in_transaction is False

        # The persist reaches the repository write transaction, both
        # INSERTs succeed, and the deterministic commit failure
        # propagates.
        with pytest.raises(sqlite3.OperationalError, match="deterministic commit failure"):
            persist_fixed(conn, message_id=5, update_id=1005)

        # The repository rolled its own transaction back: nothing is
        # pending and neither inserted row remains visible.
        assert conn.in_transaction is False
        assert count_transactions(conn) == 0
        assert count_processed_updates(conn) == 0
        assert transaction_row(conn, message_id=5) is None
        assert processed_update_row(conn, update_id=1005) is None

        # The connection remains usable after the failed persist.
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0
    finally:
        conn.close()

    # The on-disk database also contains neither row: the failed commit
    # left no repository data behind.
    verify = sqlite3.connect(db_path)
    try:
        assert int(verify.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]) == 0
        assert (
            int(verify.execute("SELECT COUNT(*) FROM processed_updates").fetchone()[0]) == 0
        )
    finally:
        verify.close()


# ---------------------------------------------------------------------------
# active caller transaction guard
# ---------------------------------------------------------------------------


def test_persist_rejects_active_caller_transaction_without_touching_it() -> None:
    conn = open_memory()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO processed_updates ("
            " update_id, chat_id, message_thread_id, message_id, processed_at"
            ") VALUES (?, ?, ?, ?, ?)",
            (1008, -100200, 7, 998, "2026-09-01T09:00:00+00:00"),
        )
        assert conn.in_transaction is True

        with pytest.raises(RepositoryTransactionError):
            persist_fixed(conn, message_id=8, update_id=1009)

        # The repository neither committed nor rolled back the caller's
        # transaction: it is still open and the caller's row is still
        # pending inside it.
        assert conn.in_transaction is True
        assert count_processed_updates(conn) == 1
        assert count_transactions(conn) == 0

        conn.rollback()
        assert count_processed_updates(conn) == 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# serialization round-trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("amount", DECIMAL_CASES)
def test_decimal_round_trips_exactly(amount: str) -> None:
    conn = open_memory()
    try:
        persist_fixed(
            conn,
            message_id=1,
            update_id=1000,
            transaction=make_transaction(amount_usdt=Decimal(amount)),
        )
        loaded = get_transaction(conn, "1")
        assert loaded is not None
        assert loaded.amount_usdt == Decimal(amount)
        assert str(loaded.amount_usdt) == amount
        assert loaded.amount_usdt.as_tuple() == Decimal(amount).as_tuple()
    finally:
        conn.close()


@pytest.mark.parametrize("amount", DECIMAL_CASES)
def test_amount_usdt_is_stored_as_text(amount: str) -> None:
    conn = open_memory()
    try:
        persist_fixed(
            conn,
            message_id=1,
            update_id=1000,
            transaction=make_transaction(amount_usdt=Decimal(amount)),
        )
        row = conn.execute(
            "SELECT typeof(amount_usdt), amount_usdt FROM transactions"
        ).fetchone()
        assert row is not None
        assert row[0] == "text"
        assert row[1] == amount
    finally:
        conn.close()


def test_transaction_date_round_trips() -> None:
    conn = open_memory()
    try:
        persist_fixed(
            conn, message_id=1, update_id=1000,
            transaction=make_transaction(transaction_date=date(2024, 2, 29)),
        )
        loaded = get_transaction(conn, "1")
        assert loaded is not None
        assert loaded.transaction_date == date(2024, 2, 29)
    finally:
        conn.close()


def test_utc_datetimes_round_trip() -> None:
    conn = open_memory()
    try:
        persist_fixed(conn, message_id=1, update_id=1000)
        loaded = get_transaction(conn, "1")
        assert loaded is not None
        assert loaded.created_at == FIXED_CREATED_AT
        assert loaded.created_at.utcoffset() == timedelta(0)
        assert loaded.updated_at == FIXED_UPDATED_AT
        assert loaded.updated_at.utcoffset() == timedelta(0)
    finally:
        conn.close()


def test_non_utc_fixed_offset_datetimes_round_trip() -> None:
    conn = open_memory()
    try:
        created = datetime(2026, 9, 1, 15, 45, 30, tzinfo=NON_UTC_TZ)
        updated = datetime(2026, 9, 1, 16, 45, 30, tzinfo=NON_UTC_TZ)
        persist_fixed(
            conn,
            message_id=1,
            update_id=1000,
            transaction=make_transaction(created_at=created, updated_at=updated),
        )
        loaded = get_transaction(conn, "1")
        assert loaded is not None
        assert loaded.created_at == created
        assert loaded.created_at.utcoffset() == timedelta(hours=5, minutes=30)
        assert loaded.updated_at == updated
        assert loaded.updated_at.utcoffset() == timedelta(hours=5, minutes=30)
    finally:
        conn.close()


def test_deleted_at_none_round_trips() -> None:
    conn = open_memory()
    try:
        persist_fixed(conn, message_id=1, update_id=1000)
        loaded = get_transaction(conn, "1")
        assert loaded is not None
        assert loaded.status is TransactionStatus.ACTIVE
        assert loaded.deleted_at is None
    finally:
        conn.close()


def test_deleted_at_aware_datetime_round_trips() -> None:
    conn = open_memory()
    try:
        persist_fixed(
            conn,
            message_id=1,
            update_id=1000,
            transaction=make_transaction(
                status=TransactionStatus.DELETED, deleted_at=FIXED_DELETED_AT
            ),
        )
        loaded = get_transaction(conn, "1")
        assert loaded is not None
        assert loaded.status is TransactionStatus.DELETED
        assert loaded.deleted_at == FIXED_DELETED_AT
        assert loaded.deleted_at is not None
        assert loaded.deleted_at.utcoffset() == timedelta(hours=5, minutes=30)
    finally:
        conn.close()


@pytest.mark.parametrize("comment", [None, "note with spaces and digits 123"])
def test_comment_round_trips(comment: str | None) -> None:
    conn = open_memory()
    try:
        persist_fixed(
            conn, message_id=1, update_id=1000, transaction=make_transaction(comment=comment)
        )
        loaded = get_transaction(conn, "1")
        assert loaded is not None
        assert loaded.comment == comment
    finally:
        conn.close()


def test_multi_word_source_round_trips() -> None:
    conn = open_memory()
    try:
        persist_fixed(
            conn,
            message_id=1,
            update_id=1000,
            transaction=make_transaction(source="Bank Transfer Card"),
        )
        loaded = get_transaction(conn, "1")
        assert loaded is not None
        assert loaded.source == "Bank Transfer Card"
    finally:
        conn.close()


def test_direction_round_trips() -> None:
    conn = open_memory()
    try:
        persist_fixed(
            conn,
            message_id=1,
            update_id=1000,
            transaction=make_transaction(direction=Direction.INCOME),
        )
        loaded = get_transaction(conn, "1")
        assert loaded is not None
        assert loaded.direction is Direction.INCOME
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# get_transaction
# ---------------------------------------------------------------------------


def test_get_transaction_returns_persisted_transaction() -> None:
    conn = open_memory()
    try:
        result = persist_fixed(conn, message_id=5, update_id=1005)
        loaded = get_transaction(conn, "1")
        assert loaded is not None
        assert loaded == result
        assert loaded.transaction_id == "1"
    finally:
        conn.close()


def test_get_transaction_valid_id_absent_returns_none() -> None:
    conn = open_memory()
    try:
        persist_fixed(conn, message_id=1, update_id=1000)
        assert get_transaction(conn, "25") is None
        assert get_transaction(conn, "1") is not None
    finally:
        conn.close()


@pytest.mark.parametrize(
    "bad_id",
    ["", "0", "-1", "+1", "1.0", "abc", " 1", "1 ", "01", "1e3"],
)
def test_get_transaction_rejects_malformed_ids(bad_id: str) -> None:
    conn = open_memory()
    try:
        with pytest.raises(ValueError):
            get_transaction(conn, bad_id)
    finally:
        conn.close()


@pytest.mark.parametrize("bad_id", [None, 17, True, 1.5])
def test_get_transaction_rejects_non_string_ids(bad_id: Any) -> None:
    conn = open_memory()
    try:
        with pytest.raises(TypeError):
            get_transaction(conn, bad_id)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# find_transaction_by_message
# ---------------------------------------------------------------------------


def test_find_transaction_by_message_exact_hit() -> None:
    conn = open_memory()
    try:
        result = persist_fixed(conn, chat_id=-100200, message_id=5, update_id=1005)
        found = find_transaction_by_message(
            conn, TelegramMessageIdentity(chat_id=-100200, message_id=5)
        )
        assert found == result
    finally:
        conn.close()


def test_find_transaction_by_message_absent() -> None:
    conn = open_memory()
    try:
        persist_fixed(conn, chat_id=-100200, message_id=5, update_id=1005)
        assert (
            find_transaction_by_message(
                conn, TelegramMessageIdentity(chat_id=-100200, message_id=6)
            )
            is None
        )
        assert (
            find_transaction_by_message(
                conn, TelegramMessageIdentity(chat_id=-100201, message_id=5)
            )
            is None
        )
    finally:
        conn.close()


@pytest.mark.parametrize(
    "bad_identity",
    [None, 5, "identity", TelegramUpdateIdentity(update_id=1005), make_ref()],
)
def test_find_transaction_by_message_rejects_wrong_identity_type(bad_identity: Any) -> None:
    conn = open_memory()
    try:
        with pytest.raises(TypeError):
            find_transaction_by_message(conn, bad_identity)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# find_transaction_by_update
# ---------------------------------------------------------------------------


def test_find_transaction_by_update_hit() -> None:
    conn = open_memory()
    try:
        result = persist_fixed(conn, chat_id=-100200, message_id=5, update_id=1005)
        found = find_transaction_by_update(conn, TelegramUpdateIdentity(update_id=1005))
        assert found == result
    finally:
        conn.close()


def test_find_transaction_by_update_absent() -> None:
    conn = open_memory()
    try:
        persist_fixed(conn, message_id=5, update_id=1005)
        assert find_transaction_by_update(conn, TelegramUpdateIdentity(update_id=9999)) is None
    finally:
        conn.close()


@pytest.mark.parametrize(
    "bad_identity",
    [None, 5, "identity", TelegramMessageIdentity(chat_id=-100200, message_id=5), make_ref()],
)
def test_find_transaction_by_update_rejects_wrong_identity_type(bad_identity: Any) -> None:
    conn = open_memory()
    try:
        with pytest.raises(TypeError):
            find_transaction_by_update(conn, bad_identity)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# is_update_processed
# ---------------------------------------------------------------------------


def test_is_update_processed_true_after_persist() -> None:
    conn = open_memory()
    try:
        persist_fixed(conn, message_id=5, update_id=1005)
        assert is_update_processed(conn, TelegramUpdateIdentity(update_id=1005)) is True
    finally:
        conn.close()


def test_is_update_processed_false_for_unknown_update() -> None:
    conn = open_memory()
    try:
        persist_fixed(conn, message_id=5, update_id=1005)
        assert is_update_processed(conn, TelegramUpdateIdentity(update_id=9999)) is False
    finally:
        conn.close()


def test_is_update_processed_uses_processed_updates_not_transactions() -> None:
    conn = open_memory()
    try:
        # A transactions row without a matching processed_updates row
        # must not be reported as a processed update.
        insert_raw_transaction(conn, message_id=900, update_id=9000)
        assert transaction_row(conn, message_id=900) is not None
        assert is_update_processed(conn, TelegramUpdateIdentity(update_id=9000)) is False
    finally:
        conn.close()


@pytest.mark.parametrize(
    "bad_identity",
    [None, 5, "identity", TelegramMessageIdentity(chat_id=-100200, message_id=5), make_ref()],
)
def test_is_update_processed_rejects_wrong_identity_type(bad_identity: Any) -> None:
    conn = open_memory()
    try:
        with pytest.raises(TypeError):
            is_update_processed(conn, bad_identity)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# corrupt stored rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "extra"),
    [
        ("amount_usdt", "abc", {}),
        ("transaction_date", "not-a-date", {}),
        ("created_at", "2026-09-01T10:00:00", {}),
        ("deleted_at", "garbage", {"status": "deleted"}),
    ],
)
def test_corrupt_rows_raise_repository_data_error(
    field: str, value: str, extra: dict[str, str]
) -> None:
    conn = open_memory()
    try:
        overrides: dict[str, Any] = {field: value}
        overrides.update(extra)
        insert_raw_transaction(conn, **overrides)
        with pytest.raises(RepositoryDataError) as excinfo:
            get_transaction(conn, "1")
        assert excinfo.value.__cause__ is not None
    finally:
        conn.close()


def test_find_transaction_by_update_corrupt_row_raises() -> None:
    conn = open_memory()
    try:
        insert_raw_transaction(conn, amount_usdt="abc", message_id=900, update_id=9000)
        with pytest.raises(RepositoryDataError):
            find_transaction_by_update(conn, TelegramUpdateIdentity(update_id=9000))
    finally:
        conn.close()


def test_corrupt_comment_blob_raises_repository_data_error_with_cause() -> None:
    conn = open_memory()
    try:
        # Direct SQL stores a BLOB in the nullable comment TEXT column.
        insert_raw_transaction(conn, comment=b"\x00\x01\x02")
        row = transaction_row(conn, message_id=900)
        assert row is not None
        assert isinstance(row[5], bytes)
        with pytest.raises(RepositoryDataError) as excinfo:
            get_transaction(conn, "1")
        assert excinfo.value.__cause__ is not None
        assert isinstance(excinfo.value.__cause__, TypeError)
    finally:
        conn.close()


def test_corrupt_amount_usdt_blob_raises_repository_data_error_with_cause() -> None:
    conn = open_memory()
    try:
        # Direct SQL stores a BLOB in the NOT NULL amount_usdt TEXT column.
        insert_raw_transaction(conn, amount_usdt=b"25")
        row = transaction_row(conn, message_id=900)
        assert row is not None
        assert isinstance(row[2], bytes)
        with pytest.raises(RepositoryDataError) as excinfo:
            get_transaction(conn, "1")
        assert excinfo.value.__cause__ is not None
        assert isinstance(excinfo.value.__cause__, TypeError)
    finally:
        conn.close()


def test_corrupt_source_blob_raises_repository_data_error_with_cause() -> None:
    conn = open_memory()
    try:
        insert_raw_transaction(conn, source=b"card")
        with pytest.raises(RepositoryDataError) as excinfo:
            find_transaction_by_message(
                conn, TelegramMessageIdentity(chat_id=-100200, message_id=900)
            )
        assert excinfo.value.__cause__ is not None
        assert isinstance(excinfo.value.__cause__, TypeError)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# time / side-effect safety
# ---------------------------------------------------------------------------


def test_processed_at_is_stored_exactly_and_independent_of_transaction_times() -> None:
    conn = open_memory()
    try:
        result = persist_fixed(conn, message_id=5, update_id=1005)
        row = processed_update_row(conn, update_id=1005)
        assert row is not None
        assert row[4] == FIXED_PROCESSED_AT.isoformat()
        assert row[4] != result.created_at.isoformat()
        assert row[4] != result.updated_at.isoformat()
    finally:
        conn.close()


def test_processed_at_accepts_non_utc_offset() -> None:
    conn = open_memory()
    try:
        processed_at = datetime(2026, 9, 1, 17, 45, 0, tzinfo=NON_UTC_TZ)
        persist_fixed(conn, message_id=5, update_id=1005, processed_at=processed_at)
        row = processed_update_row(conn, update_id=1005)
        assert row is not None
        assert row[4] == processed_at.isoformat()
        assert row[4].endswith("+05:30")
    finally:
        conn.close()


def test_read_functions_do_not_alter_database_content() -> None:
    conn = open_memory()
    try:
        persist_fixed(conn, message_id=5, update_id=1005)
        before_transactions = conn.execute(
            "SELECT * FROM transactions ORDER BY id"
        ).fetchall()
        before_updates = conn.execute(
            "SELECT * FROM processed_updates ORDER BY update_id"
        ).fetchall()
        assert conn.in_transaction is False

        get_transaction(conn, "1")
        get_transaction(conn, "999")
        find_transaction_by_message(
            conn, TelegramMessageIdentity(chat_id=-100200, message_id=5)
        )
        find_transaction_by_message(conn, TelegramMessageIdentity(chat_id=-1, message_id=1))
        find_transaction_by_update(conn, TelegramUpdateIdentity(update_id=1005))
        find_transaction_by_update(conn, TelegramUpdateIdentity(update_id=1))
        is_update_processed(conn, TelegramUpdateIdentity(update_id=1005))
        is_update_processed(conn, TelegramUpdateIdentity(update_id=1))

        assert conn.in_transaction is False
        after_transactions = conn.execute(
            "SELECT * FROM transactions ORDER BY id"
        ).fetchall()
        after_updates = conn.execute(
            "SELECT * FROM processed_updates ORDER BY update_id"
        ).fetchall()
        assert after_transactions == before_transactions
        assert after_updates == before_updates
    finally:
        conn.close()


@pytest.mark.parametrize(
    "forbidden",
    [
        "os.environ",
        "getenv",
        "datetime.now",
        "utcnow",
        "time.time",
        "date.today",
        "today()",
        "perf_counter",
        "monotonic",
        "sqlite3.connect",
        "user_version",
        "CREATE TABLE",
        "ALTER TABLE",
        "PRAGMA",
        "import os",
        "import time",
    ],
)
def test_repository_source_avoids_forbidden_operations(forbidden: str) -> None:
    assert forbidden not in REPOSITORY_SOURCE


def test_repository_public_api_is_small_and_deliberate() -> None:
    assert repository_module.__all__ == [
        "RepositoryDataError",
        "RepositoryTransactionError",
        "find_transaction_by_message",
        "find_transaction_by_update",
        "get_transaction",
        "is_update_processed",
        "persist_transaction",
    ]
    for name in repository_module.__all__:
        assert callable(getattr(repository_module, name))
