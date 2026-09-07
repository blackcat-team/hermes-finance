"""Tests for the amount-only correction orchestration facade (stage H1).

All tests are deterministic: no network, no wall-clock time, no
randomness, no environment access. The facade is exercised through the
real accepted repository read, the accepted D2 mutation core, and the
real SQLite schema; no lower-layer business logic is re-implemented or
stubbed here.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import hermes_finance
import hermes_finance.corrections as corrections_module
from hermes_finance import (
    Direction,
    FinanceConfig,
    IngestDisposition,
    TelegramMessageRef,
    Transaction,
    TransactionNotActiveError,
    TransactionNotFoundError,
    TransactionStatus,
    edit_transaction,
    edit_transaction_amount,
    ingest_transaction,
    open_database,
    soft_delete_transaction,
)
from hermes_finance.provenance import TelegramUpdateIdentity
from hermes_finance.repository import get_transaction, is_update_processed

CORRECTIONS_SOURCE: str = Path(corrections_module.__file__).read_text(encoding="utf-8")

FIXED_TZ = UTC
NON_UTC_TZ = timezone(timedelta(hours=3))

FIXED_DATE = date(2026, 9, 1)
FIXED_CREATED_AT = datetime(2026, 9, 1, 10, 0, 0, tzinfo=FIXED_TZ)
FIXED_PROCESSED_AT = datetime(2026, 9, 1, 12, 15, 0, tzinfo=FIXED_TZ)
EDIT_UPDATED_AT = datetime(2026, 9, 2, 9, 30, 0, tzinfo=NON_UTC_TZ)
DELETE_T1 = datetime(2026, 9, 3, 8, 0, 0, tzinfo=FIXED_TZ)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def open_memory() -> sqlite3.Connection:
    """Open a fresh migrated in-memory database connection."""
    return open_database(
        FinanceConfig(database_path=Path(":memory:"), business_timezone=FIXED_TZ)
    )


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
        "amount_usdt": Decimal(10),
        "category": "Инфраструктура",
        "source": "Сервер B",
        "comment": "original note",
        "transaction_date": FIXED_DATE,
        "created_at": FIXED_CREATED_AT,
        "updated_at": FIXED_CREATED_AT,
    }
    values.update(overrides)
    return Transaction(**values)


def create_active(
    conn: sqlite3.Connection,
    *,
    message_id: int = 1,
    update_id: int = 1000,
    transaction: Transaction | None = None,
) -> Transaction:
    """Create one persisted transaction through the accepted ingest service."""
    result = ingest_transaction(
        conn,
        transaction if transaction is not None else make_transaction(),
        make_ref(message_id=message_id, update_id=update_id),
        processed_at=FIXED_PROCESSED_AT,
    )
    assert result.disposition is IngestDisposition.CREATED
    return result.transaction


def created_id(created: Transaction) -> str:
    """The repository ID of a created transaction (must be assigned)."""
    assert created.transaction_id is not None
    return created.transaction_id


def raw_row(conn: sqlite3.Connection, transaction_id: str) -> Any:
    """The full transactions row for the given repository ID, or None."""
    return conn.execute(
        "SELECT * FROM transactions WHERE id = ?", (int(transaction_id),)
    ).fetchone()


def processed_updates_snapshot(conn: sqlite3.Connection) -> Any:
    """The full logical snapshot of the processed_updates table."""
    return conn.execute("SELECT * FROM processed_updates ORDER BY update_id").fetchall()


# ---------------------------------------------------------------------------
# the amount-only composition
# ---------------------------------------------------------------------------


def test_amount_changes_exactly_and_only() -> None:
    conn = open_memory()
    try:
        created = create_active(conn)  # -10 Инфраструктура Сервер B
        tx_id = created_id(created)

        result = edit_transaction_amount(
            conn, tx_id, amount_usdt="12", updated_at=EDIT_UPDATED_AT
        )

        assert result.amount_usdt == Decimal(12)
        assert result.amount_usdt.as_tuple() == Decimal(12).as_tuple()
        row = raw_row(conn, tx_id)
        assert row is not None
        assert row[2] == "12"
    finally:
        conn.close()


def test_direction_preserved_for_expense() -> None:
    """An existing ``-10`` expense edited to ``12`` remains an expense."""
    conn = open_memory()
    try:
        created = create_active(conn)  # expense
        tx_id = created_id(created)

        result = edit_transaction_amount(
            conn, tx_id, amount_usdt=12, updated_at=EDIT_UPDATED_AT
        )

        assert result.direction is Direction.EXPENSE
        row = raw_row(conn, tx_id)
        assert row is not None
        assert row[1] == "expense"
    finally:
        conn.close()


def test_direction_preserved_for_income() -> None:
    conn = open_memory()
    try:
        created = create_active(
            conn, transaction=make_transaction(direction=Direction.INCOME)
        )
        tx_id = created_id(created)

        result = edit_transaction_amount(
            conn, tx_id, amount_usdt=Decimal("0.5"), updated_at=EDIT_UPDATED_AT
        )

        assert result.direction is Direction.INCOME
        row = raw_row(conn, tx_id)
        assert row is not None
        assert row[1] == "income"
    finally:
        conn.close()


def test_every_other_business_field_is_passed_through_unchanged() -> None:
    conn = open_memory()
    try:
        created = create_active(conn)
        tx_id = created_id(created)

        result = edit_transaction_amount(
            conn, tx_id, amount_usdt="12", updated_at=EDIT_UPDATED_AT
        )

        assert result.category == created.category
        assert result.source == created.source
        assert result.comment == created.comment
        assert result.transaction_date == created.transaction_date
        assert result.transaction_id == tx_id
        assert result.created_at == created.created_at
        assert result.status is TransactionStatus.ACTIVE
        assert result.deleted_at is None
    finally:
        conn.close()


def test_updated_at_equals_the_caller_snapshot_exactly() -> None:
    conn = open_memory()
    try:
        created = create_active(conn)
        tx_id = created_id(created)

        result = edit_transaction_amount(
            conn, tx_id, amount_usdt="12", updated_at=EDIT_UPDATED_AT
        )

        assert result.updated_at == EDIT_UPDATED_AT
        assert result.updated_at.utcoffset() == timedelta(hours=3)
        row = raw_row(conn, tx_id)
        assert row is not None
        assert row[8] == EDIT_UPDATED_AT.isoformat()
    finally:
        conn.close()


def test_provenance_and_processed_updates_preserved() -> None:
    conn = open_memory()
    try:
        created = create_active(conn, message_id=5, update_id=1005)
        tx_id = created_id(created)
        row_before = raw_row(conn, tx_id)
        updates_before = processed_updates_snapshot(conn)

        edit_transaction_amount(conn, tx_id, amount_usdt="12", updated_at=EDIT_UPDATED_AT)

        row_after = raw_row(conn, tx_id)
        assert row_after is not None
        assert row_after is not None and row_before is not None
        # Identity and Telegram provenance columns are untouched.
        assert row_after[0] == row_before[0]
        assert row_after[11:15] == row_before[11:15]
        # The Telegram delivery stays processed exactly as before.
        assert processed_updates_snapshot(conn) == updates_before
        assert (
            is_update_processed(conn, TelegramUpdateIdentity(update_id=1005)) is True
        )
    finally:
        conn.close()


def test_loaded_transaction_is_not_mutated() -> None:
    conn = open_memory()
    try:
        created = create_active(conn)
        tx_id = created_id(created)
        loaded_before = get_transaction(conn, tx_id)
        assert loaded_before is not None

        edit_transaction_amount(conn, tx_id, amount_usdt="12", updated_at=EDIT_UPDATED_AT)

        # The earlier loaded object still shows the pre-edit amount: the
        # facade never mutates a loaded Transaction in place.
        assert loaded_before.amount_usdt == Decimal(10)
        assert created.amount_usdt == Decimal(10)
    finally:
        conn.close()


def test_result_matches_a_direct_full_replacement_edit() -> None:
    """The facade is exactly edit_transaction with the current fields."""
    conn = open_memory()
    try:
        create_active(conn, message_id=1, update_id=1000)
        create_active(
            conn,
            message_id=2,
            update_id=1001,
            transaction=make_transaction(
                direction=Direction.INCOME,
                amount_usdt=Decimal("7.25"),
                category="Работа",
                source="Проект A",
                comment=None,
                transaction_date=date(2026, 8, 20),
            ),
        )

        via_facade = edit_transaction_amount(
            conn, "2", amount_usdt="3.5", updated_at=EDIT_UPDATED_AT
        )

        current = get_transaction(conn, "2")
        assert current is not None
        via_direct = edit_transaction(
            conn,
            "1",
            direction=current.direction,
            amount_usdt=Decimal("3.5"),
            category=current.category,
            source=current.source,
            comment=current.comment,
            transaction_date=current.transaction_date,
            updated_at=EDIT_UPDATED_AT,
        )

        assert via_facade.amount_usdt == via_direct.amount_usdt
        assert via_facade.direction == via_direct.direction
        assert via_facade.updated_at == via_direct.updated_at
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# missing / deleted / invalid targets and amounts
# ---------------------------------------------------------------------------


def test_missing_id_raises_transaction_not_found() -> None:
    conn = open_memory()
    try:
        with pytest.raises(TransactionNotFoundError):
            edit_transaction_amount(
                conn, "25", amount_usdt="12", updated_at=EDIT_UPDATED_AT
            )
        assert conn.in_transaction is False
    finally:
        conn.close()


def test_deleted_target_rejects_edit_without_writes() -> None:
    conn = open_memory()
    try:
        created = create_active(conn)
        tx_id = created_id(created)
        soft_delete_transaction(conn, tx_id, deleted_at=DELETE_T1)
        row_before = raw_row(conn, tx_id)
        updates_before = processed_updates_snapshot(conn)

        with pytest.raises(TransactionNotActiveError):
            edit_transaction_amount(
                conn, tx_id, amount_usdt="12", updated_at=EDIT_UPDATED_AT
            )

        assert raw_row(conn, tx_id) == row_before
        assert processed_updates_snapshot(conn) == updates_before
        assert conn.in_transaction is False
    finally:
        conn.close()


@pytest.mark.parametrize(
    "bad_id",
    ["", "0", "-1", "+1", "1.0", "abc", "01", " 1"],
)
def test_invalid_id_rejects_edit(bad_id: str) -> None:
    conn = open_memory()
    try:
        created = create_active(conn)
        tx_id = created_id(created)
        row_before = raw_row(conn, tx_id)

        with pytest.raises(ValueError):
            edit_transaction_amount(
                conn, bad_id, amount_usdt="12", updated_at=EDIT_UPDATED_AT
            )

        assert raw_row(conn, tx_id) == row_before
    finally:
        conn.close()


@pytest.mark.parametrize("bad_id", [17, None, True, 1.5])
def test_non_string_id_rejects_edit(bad_id: Any) -> None:
    conn = open_memory()
    try:
        with pytest.raises(TypeError):
            edit_transaction_amount(
                conn, bad_id, amount_usdt="12", updated_at=EDIT_UPDATED_AT
            )
    finally:
        conn.close()


@pytest.mark.parametrize(
    "bad_amount",
    ["abc", "12,5", "1.2.3", "0", "-5", 0, -12, 25.5, True, None],
)
def test_invalid_amount_rejects_edit_without_writes(bad_amount: Any) -> None:
    conn = open_memory()
    try:
        created = create_active(conn)
        tx_id = created_id(created)
        row_before = raw_row(conn, tx_id)
        updates_before = processed_updates_snapshot(conn)

        with pytest.raises((TypeError, ValueError)):
            edit_transaction_amount(
                conn, tx_id, amount_usdt=bad_amount, updated_at=EDIT_UPDATED_AT
            )

        assert raw_row(conn, tx_id) == row_before
        assert processed_updates_snapshot(conn) == updates_before
        assert conn.in_transaction is False
    finally:
        conn.close()


def test_negative_amount_does_not_change_direction() -> None:
    """A negative magnitude is rejected, never interpreted as a flip."""
    conn = open_memory()
    try:
        created = create_active(conn)  # expense
        tx_id = created_id(created)
        row_before = raw_row(conn, tx_id)

        with pytest.raises(ValueError):
            edit_transaction_amount(
                conn, tx_id, amount_usdt="-12", updated_at=EDIT_UPDATED_AT
            )

        # The stored expense row is unchanged: no silent direction flip.
        assert raw_row(conn, tx_id) == row_before
        assert conn.in_transaction is False
    finally:
        conn.close()


@pytest.mark.parametrize(
    "bad_updated_at",
    [datetime(2026, 9, 2, 9, 30), "2026-09-02T09:30:00+00:00", None],  # noqa: DTZ001 - naive is the point
)
def test_invalid_updated_at_rejects_edit(bad_updated_at: Any) -> None:
    conn = open_memory()
    try:
        created = create_active(conn)
        tx_id = created_id(created)
        row_before = raw_row(conn, tx_id)

        with pytest.raises((TypeError, ValueError)):
            edit_transaction_amount(
                conn, tx_id, amount_usdt="12", updated_at=bad_updated_at
            )

        assert raw_row(conn, tx_id) == row_before
    finally:
        conn.close()


def test_missing_target_rejects_before_any_write() -> None:
    """A missing ID fails in the read step, before the mutation begins."""
    conn = open_memory()
    try:
        create_active(conn)
        updates_before = processed_updates_snapshot(conn)

        with pytest.raises(TransactionNotFoundError):
            edit_transaction_amount(
                conn, "99", amount_usdt="12", updated_at=EDIT_UPDATED_AT
            )

        assert processed_updates_snapshot(conn) == updates_before
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# connection contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_connection", ["not-a-connection", None, 5, object()])
def test_edit_transaction_amount_rejects_non_connection(bad_connection: Any) -> None:
    with pytest.raises(TypeError):
        edit_transaction_amount(
            bad_connection, "1", amount_usdt="12", updated_at=EDIT_UPDATED_AT
        )


# ---------------------------------------------------------------------------
# time / side-effect safety and public API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    [
        "os.environ",
        "getenv",
        "datetime.now",
        "utcnow",
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
        "SELECT",
        "INSERT INTO",
        "UPDATE ",
        "DELETE FROM",
        ".execute",
    ],
)
def test_corrections_source_avoids_forbidden_operations(forbidden: str) -> None:
    assert forbidden not in CORRECTIONS_SOURCE


def test_corrections_public_api_is_small_and_deliberate() -> None:
    assert corrections_module.__all__ == ["edit_transaction_amount"]
    assert callable(corrections_module.edit_transaction_amount)


def test_package_exports_correction_api() -> None:
    assert "edit_transaction_amount" in hermes_finance.__all__
    assert hermes_finance.edit_transaction_amount is edit_transaction_amount
