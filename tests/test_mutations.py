"""Tests for the deterministic transaction mutation operations (stage D2).

All tests are deterministic: no network, no wall-clock time, no
randomness, no environment access. Filesystem mutation happens only
inside pytest-owned temporary directories or in in-memory SQLite
databases.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

import pytest

import hermes_finance
import hermes_finance.mutations as mutations_module
from hermes_finance import (
    Direction,
    FinanceConfig,
    IngestDisposition,
    TelegramMessageRef,
    TelegramUpdateIdentity,
    Transaction,
    TransactionStatus,
    edit_transaction,
    ingest_transaction,
    list_recent_transactions,
    list_transactions_by_date,
    list_transactions_by_month,
    open_database,
    soft_delete_transaction,
)
from hermes_finance.mutations import (
    TransactionMutationError,
    TransactionNotActiveError,
    TransactionNotFoundError,
)
from hermes_finance.repository import (
    RepositoryDataError,
    RepositoryTransactionError,
    get_transaction,
    is_update_processed,
)

MUTATIONS_SOURCE: str = Path(mutations_module.__file__).read_text(encoding="utf-8")

FIXED_TZ = UTC
NON_UTC_TZ = timezone(timedelta(hours=5, minutes=45))

FIXED_DATE = date(2026, 9, 1)
EDITED_DATE = date(2026, 8, 15)
FIXED_CREATED_AT = datetime(2026, 9, 1, 10, 0, 0, tzinfo=FIXED_TZ)
INITIAL_UPDATED_AT = datetime(2026, 9, 1, 11, 30, 0, tzinfo=FIXED_TZ)
FIXED_PROCESSED_AT = datetime(2026, 9, 1, 12, 15, 0, tzinfo=FIXED_TZ)
EDIT_UPDATED_AT = datetime(2026, 9, 2, 9, 30, 0, tzinfo=NON_UTC_TZ)
DELETE_T1 = datetime(2026, 9, 3, 8, 0, 0, tzinfo=FIXED_TZ)
DELETE_T2 = datetime(2026, 9, 4, 18, 20, 0, tzinfo=NON_UTC_TZ)


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
        "comment": "original note",
        "transaction_date": FIXED_DATE,
        "created_at": FIXED_CREATED_AT,
        "updated_at": INITIAL_UPDATED_AT,
        "status": TransactionStatus.ACTIVE,
        "deleted_at": None,
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
    """Create one persisted transaction through the accepted ingest service.

    Using ingest (rather than a bare persist) guarantees that both the
    ``transactions`` provenance columns and the ``processed_updates``
    row contain the authoritative C3 state.
    """
    result = ingest_transaction(
        conn,
        transaction if transaction is not None else make_transaction(),
        make_ref(message_id=message_id, update_id=update_id),
        processed_at=FIXED_PROCESSED_AT,
    )
    assert result.disposition is IngestDisposition.CREATED
    return result.transaction


def edit_kwargs(**overrides: Any) -> dict[str, Any]:
    """A valid complete business-field set for edit_transaction."""
    values: dict[str, Any] = {
        "direction": Direction.INCOME,
        "amount_usdt": Decimal("123.456789"),
        "category": "Travel",
        "source": "Bank Transfer",
        "comment": "edited note",
        "transaction_date": EDITED_DATE,
        "updated_at": EDIT_UPDATED_AT,
    }
    values.update(overrides)
    return values


def count_transactions(conn: sqlite3.Connection) -> int:
    """Number of transactions rows."""
    row = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()
    return int(row[0])


def count_processed_updates(conn: sqlite3.Connection) -> int:
    """Number of processed_updates rows."""
    row = conn.execute("SELECT COUNT(*) FROM processed_updates").fetchone()
    return int(row[0])


def raw_row(conn: sqlite3.Connection, transaction_id: str) -> Any:
    """The full transactions row for the given repository ID, or None."""
    return conn.execute(
        "SELECT * FROM transactions WHERE id = ?", (int(transaction_id),)
    ).fetchone()


def provenance_of(conn: sqlite3.Connection, transaction_id: str) -> Any:
    """The identity and provenance columns of one transactions row."""
    return conn.execute(
        "SELECT id, chat_id, message_thread_id, message_id, update_id"
        " FROM transactions WHERE id = ?",
        (int(transaction_id),),
    ).fetchone()


def processed_updates_snapshot(conn: sqlite3.Connection) -> Any:
    """The full logical snapshot of the processed_updates table."""
    return conn.execute("SELECT * FROM processed_updates ORDER BY update_id").fetchall()


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


def created_id(created: Transaction) -> str:
    """The repository ID of a created transaction (must be assigned)."""
    assert created.transaction_id is not None
    return created.transaction_id


# ---------------------------------------------------------------------------
# connection contract
# ---------------------------------------------------------------------------


class _FakeConnection:
    """An object that mimics part of the connection interface."""

    in_transaction = False


BAD_CONNECTIONS: Final[list[Any]] = [
    "not-a-connection",
    None,
    5,
    object(),
    _FakeConnection(),
]


@pytest.mark.parametrize("bad_connection", BAD_CONNECTIONS)
def test_edit_transaction_rejects_non_connection(bad_connection: Any) -> None:
    with pytest.raises(TypeError):
        edit_transaction(bad_connection, "1", **edit_kwargs())


@pytest.mark.parametrize("bad_connection", BAD_CONNECTIONS)
def test_soft_delete_transaction_rejects_non_connection(bad_connection: Any) -> None:
    with pytest.raises(TypeError):
        soft_delete_transaction(bad_connection, "1", deleted_at=DELETE_T1)


# ---------------------------------------------------------------------------
# mutation error types
# ---------------------------------------------------------------------------


def test_mutation_error_hierarchy() -> None:
    assert issubclass(TransactionNotFoundError, TransactionMutationError)
    assert issubclass(TransactionNotActiveError, TransactionMutationError)
    # Mutation errors are deliberately distinct from repository and
    # SQLite failure types: repository corruption, connection-ownership
    # violations, and SQLite errors are never collapsed into them.
    for error in (
        TransactionMutationError,
        TransactionNotFoundError,
        TransactionNotActiveError,
    ):
        assert not issubclass(error, RepositoryDataError)
        assert not issubclass(error, RepositoryTransactionError)
        assert not issubclass(error, sqlite3.Error)


# ---------------------------------------------------------------------------
# edit: happy path
# ---------------------------------------------------------------------------


def test_edit_replaces_every_editable_business_field() -> None:
    conn = open_memory()
    try:
        created = create_active(conn)
        tx_id = created_id(created)
        before_row = raw_row(conn, tx_id)
        before_updates = processed_updates_snapshot(conn)
        assert before_row is not None

        result = edit_transaction(conn, tx_id, **edit_kwargs())

        assert result.transaction_id == tx_id
        assert result.direction is Direction.INCOME
        assert result.amount_usdt == Decimal("123.456789")
        assert result.amount_usdt.as_tuple() == Decimal("123.456789").as_tuple()
        assert result.category == "Travel"
        assert result.source == "Bank Transfer"
        assert result.comment == "edited note"
        assert result.transaction_date == EDITED_DATE
        assert result.updated_at == EDIT_UPDATED_AT
        assert result.updated_at.utcoffset() == timedelta(hours=5, minutes=45)
        assert result.created_at == created.created_at
        assert result.status is TransactionStatus.ACTIVE
        assert result.deleted_at is None
        assert conn.in_transaction is False

        after_row = raw_row(conn, tx_id)
        assert after_row is not None
        # Business columns replaced.
        assert after_row[1] == "income"
        assert after_row[2] == "123.456789"
        assert after_row[3] == "Travel"
        assert after_row[4] == "Bank Transfer"
        assert after_row[5] == "edited note"
        assert after_row[6] == EDITED_DATE.isoformat()
        assert after_row[8] == EDIT_UPDATED_AT.isoformat()
        # Identity, lifecycle, and provenance columns preserved.
        assert after_row[0] == before_row[0]
        assert after_row[7] == before_row[7]  # created_at
        assert after_row[9] == "active"
        assert after_row[10] is None
        assert after_row[11:15] == before_row[11:15]  # provenance

        # No second row, no processed_updates damage, read-back agrees.
        assert count_transactions(conn) == 1
        assert processed_updates_snapshot(conn) == before_updates
        assert get_transaction(conn, tx_id) == result
    finally:
        conn.close()


def test_edit_with_unchanged_business_values_updates_updated_at() -> None:
    conn = open_memory()
    try:
        created = create_active(conn)
        tx_id = created_id(created)
        before_updates = processed_updates_snapshot(conn)

        result = edit_transaction(
            conn,
            tx_id,
            direction=created.direction,
            amount_usdt=created.amount_usdt,
            category=created.category,
            source=created.source,
            comment=created.comment,
            transaction_date=created.transaction_date,
            updated_at=EDIT_UPDATED_AT,
        )

        # Same business values, same identity, same created_at.
        assert result.transaction_id == tx_id
        assert result.direction == created.direction
        assert result.amount_usdt == created.amount_usdt
        assert result.category == created.category
        assert result.source == created.source
        assert result.comment == created.comment
        assert result.transaction_date == created.transaction_date
        assert result.created_at == created.created_at
        # The caller-supplied updated_at is NOT discarded as a no-op.
        assert result.updated_at == EDIT_UPDATED_AT
        row = raw_row(conn, tx_id)
        assert row is not None
        assert row[8] == EDIT_UPDATED_AT.isoformat()
        assert processed_updates_snapshot(conn) == before_updates
        assert count_transactions(conn) == 1
    finally:
        conn.close()


def test_edit_can_clear_comment_to_none() -> None:
    conn = open_memory()
    try:
        created = create_active(conn)
        tx_id = created_id(created)
        assert created.comment == "original note"

        result = edit_transaction(
            conn,
            tx_id,
            direction=created.direction,
            amount_usdt=created.amount_usdt,
            category=created.category,
            source=created.source,
            comment=None,
            transaction_date=created.transaction_date,
            updated_at=EDIT_UPDATED_AT,
        )

        assert result.comment is None
        row = raw_row(conn, tx_id)
        assert row is not None
        assert row[5] is None
    finally:
        conn.close()


def test_edit_and_soft_delete_preserve_provenance_and_processed_updates() -> None:
    conn = open_memory()
    try:
        created = create_active(conn, message_id=5, update_id=1005)
        tx_id = created_id(created)
        provenance_before = provenance_of(conn, tx_id)
        updates_before = processed_updates_snapshot(conn)

        edit_transaction(conn, tx_id, **edit_kwargs())
        assert provenance_of(conn, tx_id) == provenance_before
        assert processed_updates_snapshot(conn) == updates_before

        soft_delete_transaction(conn, tx_id, deleted_at=DELETE_T1)
        assert provenance_of(conn, tx_id) == provenance_before
        assert processed_updates_snapshot(conn) == updates_before
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# edit: missing / deleted targets
# ---------------------------------------------------------------------------


def test_edit_missing_id_raises_transaction_not_found() -> None:
    conn = open_memory()
    try:
        with pytest.raises(TransactionNotFoundError):
            edit_transaction(conn, "25", **edit_kwargs())
        assert count_transactions(conn) == 0
        assert conn.in_transaction is False
    finally:
        conn.close()


def test_edit_deleted_transaction_raises_transaction_not_active() -> None:
    conn = open_memory()
    try:
        created = create_active(conn)
        tx_id = created_id(created)
        soft_delete_transaction(conn, tx_id, deleted_at=DELETE_T1)
        row_before = raw_row(conn, tx_id)
        updates_before = processed_updates_snapshot(conn)

        with pytest.raises(TransactionNotActiveError):
            edit_transaction(conn, tx_id, **edit_kwargs())

        # No restore, no edit of deleted history, no row damage.
        assert raw_row(conn, tx_id) == row_before
        assert processed_updates_snapshot(conn) == updates_before
        assert conn.in_transaction is False
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# edit: input validation
# ---------------------------------------------------------------------------

INVALID_EDIT_CASES: Final[list[tuple[str, dict[str, Any], type[Exception]]]] = [
    ("float amount", {"amount_usdt": 25.5}, TypeError),
    ("bool amount", {"amount_usdt": True}, TypeError),
    ("zero amount", {"amount_usdt": 0}, ValueError),
    ("negative amount", {"amount_usdt": Decimal(-5)}, ValueError),
    ("NaN amount", {"amount_usdt": Decimal("NaN")}, ValueError),
    ("unclean amount string", {"amount_usdt": "12,5"}, ValueError),
    ("invalid direction type", {"direction": 5}, TypeError),
    ("invalid direction value", {"direction": "transfer"}, ValueError),
    ("blank category", {"category": "   "}, ValueError),
    ("blank source", {"source": "\t \n"}, ValueError),
    ("non-string category", {"category": 123}, TypeError),
    (
        "datetime as transaction_date",
        {"transaction_date": datetime(2026, 8, 15, tzinfo=UTC)},
        TypeError,
    ),
    (
        "naive updated_at",
        # naive is the point of the case
        {"updated_at": datetime(2026, 9, 2, 9, 30)},  # noqa: DTZ001
        ValueError,
    ),
    ("non-datetime updated_at", {"updated_at": "2026-09-02T09:30:00+00:00"}, TypeError),
]


@pytest.mark.parametrize(
    ("label", "overrides", "expected_error"),
    INVALID_EDIT_CASES,
)
def test_edit_rejects_invalid_business_values(
    label: str, overrides: dict[str, Any], expected_error: type[Exception]
) -> None:
    conn = open_memory()
    try:
        created = create_active(conn)
        tx_id = created_id(created)
        row_before = raw_row(conn, tx_id)
        updates_before = processed_updates_snapshot(conn)

        with pytest.raises(expected_error):
            edit_transaction(conn, tx_id, **edit_kwargs(**overrides))

        # No mutation survives any invalid edit.
        assert raw_row(conn, tx_id) == row_before
        assert processed_updates_snapshot(conn) == updates_before
        assert conn.in_transaction is False
    finally:
        conn.close()


INVALID_EDIT_IDS: Final[list[tuple[Any, type[Exception]]]] = [
    ("", ValueError),
    ("0", ValueError),
    ("-1", ValueError),
    ("+1", ValueError),
    ("1.0", ValueError),
    ("abc", ValueError),
    (" 1", ValueError),
    ("1 ", ValueError),
    ("01", ValueError),
    (17, TypeError),
    (None, TypeError),
    (True, TypeError),
    (1.5, TypeError),
]


@pytest.mark.parametrize(("bad_id", "expected_error"), INVALID_EDIT_IDS)
def test_edit_rejects_invalid_transaction_id(
    bad_id: Any, expected_error: type[Exception]
) -> None:
    conn = open_memory()
    try:
        created = create_active(conn)
        tx_id = created_id(created)
        row_before = raw_row(conn, tx_id)

        with pytest.raises(expected_error):
            edit_transaction(conn, bad_id, **edit_kwargs())

        assert raw_row(conn, tx_id) == row_before
        assert conn.in_transaction is False
    finally:
        conn.close()


def test_edit_domain_validation_failure_after_begin_rolls_back() -> None:
    conn = open_memory()
    try:
        created = create_active(conn)
        tx_id = created_id(created)
        row_before = raw_row(conn, tx_id)
        updates_before = processed_updates_snapshot(conn)

        # A float amount is rejected by the authoritative domain
        # constructor while constructing the edited Transaction, after
        # the repository write transaction has begun.
        with pytest.raises(TypeError, match="float"):
            edit_transaction(conn, tx_id, **edit_kwargs(amount_usdt=25.5))

        assert conn.in_transaction is False
        assert raw_row(conn, tx_id) == row_before
        assert processed_updates_snapshot(conn) == updates_before

        # The connection remains fully usable after the rollback.
        result = edit_transaction(conn, tx_id, **edit_kwargs())
        assert result.updated_at == EDIT_UPDATED_AT
        assert result.amount_usdt == Decimal("123.456789")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# caller-owned active transaction guard
# ---------------------------------------------------------------------------


def test_edit_rejects_active_caller_transaction_without_touching_it() -> None:
    conn = open_memory()
    try:
        created = create_active(conn)
        tx_id = created_id(created)
        row_before = raw_row(conn, tx_id)

        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO processed_updates ("
            " update_id, chat_id, message_thread_id, message_id, processed_at"
            ") VALUES (?, ?, ?, ?, ?)",
            (1008, -100200, 7, 998, "2026-09-01T09:00:00+00:00"),
        )
        assert conn.in_transaction is True

        with pytest.raises(RepositoryTransactionError):
            edit_transaction(conn, tx_id, **edit_kwargs())

        # The mutation neither committed nor rolled back the caller's
        # transaction: it is still open and the pending work is intact.
        assert conn.in_transaction is True
        assert count_processed_updates(conn) == 2
        assert raw_row(conn, tx_id) == row_before

        conn.rollback()
        assert count_processed_updates(conn) == 1
        assert raw_row(conn, tx_id) == row_before
    finally:
        conn.close()


def test_soft_delete_rejects_active_caller_transaction_without_touching_it() -> None:
    conn = open_memory()
    try:
        created = create_active(conn)
        tx_id = created_id(created)
        row_before = raw_row(conn, tx_id)

        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO processed_updates ("
            " update_id, chat_id, message_thread_id, message_id, processed_at"
            ") VALUES (?, ?, ?, ?, ?)",
            (1008, -100200, 7, 998, "2026-09-01T09:00:00+00:00"),
        )
        assert conn.in_transaction is True

        with pytest.raises(RepositoryTransactionError):
            soft_delete_transaction(conn, tx_id, deleted_at=DELETE_T1)

        assert conn.in_transaction is True
        assert count_processed_updates(conn) == 2
        assert raw_row(conn, tx_id) == row_before

        conn.rollback()
        assert count_processed_updates(conn) == 1
        assert raw_row(conn, tx_id) == row_before
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# commit failure inside the repository write transaction
# ---------------------------------------------------------------------------


class _CommitFailsConnection(sqlite3.Connection):
    """A real sqlite3.Connection whose commit() fails deterministically.

    rollback() keeps its normal behaviour, so this subclass proves that
    a commit failure inside a repository mutation triggers the
    repository's rollback path and leaves no pending repository
    transaction. Test-only; the production code has no knowledge of it.
    """

    def commit(self) -> None:
        raise sqlite3.OperationalError("deterministic commit failure")


def test_edit_commit_failure_rolls_back(tmp_path: Path) -> None:
    db_path = tmp_path / "finance.sqlite"

    bootstrap = open_database(file_config(db_path))
    try:
        created = create_active(bootstrap)
        tx_id = created_id(created)
        original_row = raw_row(bootstrap, tx_id)
        original_updates = processed_updates_snapshot(bootstrap)
    finally:
        bootstrap.close()

    conn = sqlite3.connect(db_path, factory=_CommitFailsConnection)
    try:
        assert isinstance(conn, sqlite3.Connection)
        assert conn.in_transaction is False

        with pytest.raises(sqlite3.OperationalError, match="deterministic commit failure"):
            edit_transaction(conn, tx_id, **edit_kwargs())

        # The repository rolled its own transaction back: nothing is
        # pending, the original row (business fields, identity,
        # lifecycle, provenance) is unchanged, and processed_updates
        # is unchanged.
        assert conn.in_transaction is False
        assert raw_row(conn, tx_id) == original_row
        assert processed_updates_snapshot(conn) == original_updates

        # The same connection remains usable after the rollback.
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 1
    finally:
        conn.close()

    # A second file-backed connection sees the original committed state.
    verify = sqlite3.connect(db_path)
    try:
        assert raw_row(verify, tx_id) == original_row
        assert processed_updates_snapshot(verify) == original_updates
    finally:
        verify.close()


def test_soft_delete_commit_failure_rolls_back(tmp_path: Path) -> None:
    db_path = tmp_path / "finance.sqlite"

    bootstrap = open_database(file_config(db_path))
    try:
        created = create_active(bootstrap)
        tx_id = created_id(created)
        original_row = raw_row(bootstrap, tx_id)
        original_updates = processed_updates_snapshot(bootstrap)
    finally:
        bootstrap.close()

    conn = sqlite3.connect(db_path, factory=_CommitFailsConnection)
    try:
        with pytest.raises(sqlite3.OperationalError, match="deterministic commit failure"):
            soft_delete_transaction(conn, tx_id, deleted_at=DELETE_T1)

        # The row remains ACTIVE with the original updated_at, a NULL
        # deleted_at, and untouched provenance and processed_updates.
        assert conn.in_transaction is False
        row = raw_row(conn, tx_id)
        assert row is not None
        assert row[9] == "active"
        assert row[10] is None
        assert row[8] == INITIAL_UPDATED_AT.isoformat()
        assert row[11:15] == original_row[11:15]
        assert processed_updates_snapshot(conn) == original_updates

        # The same connection remains usable after the rollback.
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 1
    finally:
        conn.close()

    # A second file-backed connection observes the ACTIVE state.
    verify = sqlite3.connect(db_path)
    try:
        row = raw_row(verify, tx_id)
        assert row is not None
        assert row[9] == "active"
        assert row[10] is None
    finally:
        verify.close()


# ---------------------------------------------------------------------------
# corrupt existing rows
# ---------------------------------------------------------------------------


def test_edit_corrupt_row_raises_repository_data_error() -> None:
    conn = open_memory()
    try:
        insert_raw_transaction(conn, amount_usdt="abc")
        row_before = raw_row(conn, "1")

        with pytest.raises(RepositoryDataError) as excinfo:
            edit_transaction(conn, "1", **edit_kwargs())

        assert excinfo.value.__cause__ is not None
        # The corrupt row is neither silently repaired nor mutated.
        assert conn.in_transaction is False
        assert raw_row(conn, "1") == row_before
    finally:
        conn.close()


def test_soft_delete_corrupt_row_raises_repository_data_error() -> None:
    conn = open_memory()
    try:
        insert_raw_transaction(conn, amount_usdt="abc")
        row_before = raw_row(conn, "1")

        with pytest.raises(RepositoryDataError) as excinfo:
            soft_delete_transaction(conn, "1", deleted_at=DELETE_T1)

        assert excinfo.value.__cause__ is not None
        assert conn.in_transaction is False
        assert raw_row(conn, "1") == row_before
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# soft delete: happy path and idempotency
# ---------------------------------------------------------------------------


def test_soft_delete_active_transaction() -> None:
    conn = open_memory()
    try:
        created = create_active(conn)
        tx_id = created_id(created)
        row_before = raw_row(conn, tx_id)
        updates_before = processed_updates_snapshot(conn)
        assert row_before is not None

        result = soft_delete_transaction(conn, tx_id, deleted_at=DELETE_T1)

        # Every business field, the identity, and created_at survive.
        assert result.transaction_id == tx_id
        assert result.direction == created.direction
        assert result.amount_usdt == created.amount_usdt
        assert result.category == created.category
        assert result.source == created.source
        assert result.comment == created.comment
        assert result.transaction_date == created.transaction_date
        assert result.created_at == created.created_at
        # Lifecycle columns become DELETED with the caller timestamp.
        assert result.status is TransactionStatus.DELETED
        assert result.updated_at == DELETE_T1
        assert result.deleted_at == DELETE_T1
        assert conn.in_transaction is False

        row = raw_row(conn, tx_id)
        assert row is not None
        assert row[9] == "deleted"
        assert row[10] == DELETE_T1.isoformat()
        assert row[8] == DELETE_T1.isoformat()
        assert row[7] == row_before[7]  # created_at
        assert row[11:15] == row_before[11:15]  # provenance

        # The Telegram delivery stays processed; no row damage.
        assert processed_updates_snapshot(conn) == updates_before
        assert is_update_processed(conn, TelegramUpdateIdentity(update_id=1000)) is True

        # The low-level read retrieves the DELETED row.
        loaded = get_transaction(conn, tx_id)
        assert loaded == result
        assert loaded is not None
        assert loaded.status is TransactionStatus.DELETED

        # D1 normal queries exclude it everywhere.
        assert list_recent_transactions(conn) == ()
        assert list_transactions_by_date(conn, FIXED_DATE) == ()
        assert list_transactions_by_month(conn, year=2026, month=9) == ()
    finally:
        conn.close()


def test_repeated_soft_delete_is_idempotent() -> None:
    conn = open_memory()
    try:
        created = create_active(conn)
        tx_id = created_id(created)

        first = soft_delete_transaction(conn, tx_id, deleted_at=DELETE_T1)
        row_after_first = raw_row(conn, tx_id)
        updates_after_first = processed_updates_snapshot(conn)

        second = soft_delete_transaction(conn, tx_id, deleted_at=DELETE_T2)

        # The second delete neither raises nor rewrites any value.
        assert second.status is TransactionStatus.DELETED
        assert second == first
        assert second.deleted_at == DELETE_T1
        assert second.updated_at == DELETE_T1
        assert raw_row(conn, tx_id) == row_after_first
        assert processed_updates_snapshot(conn) == updates_after_first
        assert count_transactions(conn) == 1
        assert conn.in_transaction is False
    finally:
        conn.close()


def test_repeated_soft_delete_still_validates_deleted_at() -> None:
    conn = open_memory()
    try:
        created = create_active(conn)
        tx_id = created_id(created)
        soft_delete_transaction(conn, tx_id, deleted_at=DELETE_T1)
        row_before = raw_row(conn, tx_id)

        naive = datetime(2026, 9, 5, 10, 0, 0)  # noqa: DTZ001 - naive is the point
        with pytest.raises(ValueError):
            soft_delete_transaction(conn, tx_id, deleted_at=naive)

        assert raw_row(conn, tx_id) == row_before
        assert conn.in_transaction is False
    finally:
        conn.close()


def test_soft_delete_missing_id_raises_transaction_not_found() -> None:
    conn = open_memory()
    try:
        with pytest.raises(TransactionNotFoundError):
            soft_delete_transaction(conn, "25", deleted_at=DELETE_T1)
        assert count_transactions(conn) == 0
        assert conn.in_transaction is False
    finally:
        conn.close()


INVALID_DELETED_AT_CASES: Final[list[tuple[Any, type[Exception]]]] = [
    (datetime(2026, 9, 3, 8, 0, 0), ValueError),  # noqa: DTZ001 - naive is the point
    ("2026-09-03T08:00:00+00:00", TypeError),
    (1782000000, TypeError),
    (None, TypeError),
]


@pytest.mark.parametrize(("bad_deleted_at", "expected_error"), INVALID_DELETED_AT_CASES)
def test_soft_delete_rejects_invalid_deleted_at(
    bad_deleted_at: Any, expected_error: type[Exception]
) -> None:
    conn = open_memory()
    try:
        created = create_active(conn)
        tx_id = created_id(created)
        row_before = raw_row(conn, tx_id)

        with pytest.raises(expected_error):
            soft_delete_transaction(conn, tx_id, deleted_at=bad_deleted_at)

        assert raw_row(conn, tx_id) == row_before
        assert conn.in_transaction is False
    finally:
        conn.close()


@pytest.mark.parametrize(("bad_id", "expected_error"), INVALID_EDIT_IDS)
def test_soft_delete_rejects_invalid_transaction_id(
    bad_id: Any, expected_error: type[Exception]
) -> None:
    conn = open_memory()
    try:
        created = create_active(conn)
        tx_id = created_id(created)
        row_before = raw_row(conn, tx_id)

        with pytest.raises(expected_error):
            soft_delete_transaction(conn, bad_id, deleted_at=DELETE_T1)

        assert raw_row(conn, tx_id) == row_before
        assert conn.in_transaction is False
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# D1 visibility integration
# ---------------------------------------------------------------------------


def test_d1_visibility_after_soft_delete() -> None:
    conn = open_memory()
    try:
        created = create_active(conn)
        tx_id = created_id(created)
        assert [t.transaction_id for t in list_recent_transactions(conn)] == [tx_id]
        assert [t.transaction_id for t in list_transactions_by_date(conn, FIXED_DATE)] == [tx_id]
        assert [
            t.transaction_id
            for t in list_transactions_by_month(conn, year=2026, month=9)
        ] == [tx_id]

        soft_delete_transaction(conn, tx_id, deleted_at=DELETE_T1)

        # Low-level visibility keeps the DELETED row.
        loaded = get_transaction(conn, tx_id)
        assert loaded is not None
        assert loaded.status is TransactionStatus.DELETED

        # D1 normal queries exclude it everywhere.
        assert list_recent_transactions(conn) == ()
        assert list_transactions_by_date(conn, FIXED_DATE) == ()
        assert list_transactions_by_month(conn, year=2026, month=9) == ()
    finally:
        conn.close()


def test_edit_visibility_through_d1() -> None:
    conn = open_memory()
    try:
        created = create_active(conn)  # business date 2026-09-01
        tx_id = created_id(created)
        other = create_active(
            conn,
            message_id=2,
            update_id=1001,
            transaction=make_transaction(transaction_date=date(2026, 8, 20)),
        )
        other_id = created_id(other)

        edited = edit_transaction(conn, tx_id, **edit_kwargs())
        assert edited.transaction_date == EDITED_DATE

        # The old date no longer includes it; the new date includes the
        # same transaction ID (no duplicate row was created).
        assert list_transactions_by_date(conn, FIXED_DATE) == ()
        assert [t.transaction_id for t in list_transactions_by_date(conn, EDITED_DATE)] == [tx_id]

        # Month queries follow the edited business date.
        assert [
            t.transaction_id
            for t in list_transactions_by_month(conn, year=2026, month=8)
        ] == [other_id, tx_id]
        assert list_transactions_by_month(conn, year=2026, month=9) == ()

        # Recent ordering follows the edited business date.
        assert [t.transaction_id for t in list_recent_transactions(conn, limit=10)] == [
            other_id,
            tx_id,
        ]

        assert count_transactions(conn) == 2
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# ingest regression around a DELETED transaction
# ---------------------------------------------------------------------------


def test_replay_after_soft_delete_returns_existing_deleted_transaction() -> None:
    conn = open_memory()
    try:
        created = create_active(conn)
        tx_id = created_id(created)
        soft_delete_transaction(conn, tx_id, deleted_at=DELETE_T1)

        # The delivery stays processed, so the exact original update
        # replays as DUPLICATE_UPDATE and resolves the existing (now
        # DELETED) logical transaction. Ingest is not modified: no
        # resurrection, no second row.
        replay = ingest_transaction(
            conn,
            make_transaction(),
            make_ref(message_id=1, update_id=1000),
            processed_at=FIXED_PROCESSED_AT,
        )
        assert replay.disposition is IngestDisposition.DUPLICATE_UPDATE
        assert replay.transaction.transaction_id == tx_id
        assert replay.transaction.status is TransactionStatus.DELETED
        assert replay.transaction.deleted_at == DELETE_T1

        assert count_transactions(conn) == 1
        assert count_processed_updates(conn) == 1
        loaded = get_transaction(conn, tx_id)
        assert loaded is not None
        assert loaded.status is TransactionStatus.DELETED
    finally:
        conn.close()


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
    ],
)
def test_mutations_source_avoids_forbidden_operations(forbidden: str) -> None:
    assert forbidden not in MUTATIONS_SOURCE


def test_mutations_public_api_is_small_and_deliberate() -> None:
    assert mutations_module.__all__ == [
        "TransactionMutationError",
        "TransactionNotActiveError",
        "TransactionNotFoundError",
        "edit_transaction",
        "soft_delete_transaction",
    ]
    for name in mutations_module.__all__:
        assert callable(getattr(mutations_module, name))


def test_package_exports_mutation_api() -> None:
    for name in (
        "TransactionMutationError",
        "TransactionNotFoundError",
        "TransactionNotActiveError",
        "edit_transaction",
        "soft_delete_transaction",
    ):
        assert name in hermes_finance.__all__
    assert hermes_finance.edit_transaction is edit_transaction
    assert hermes_finance.soft_delete_transaction is soft_delete_transaction
    assert hermes_finance.TransactionMutationError is TransactionMutationError
    assert hermes_finance.TransactionNotFoundError is TransactionNotFoundError
    assert hermes_finance.TransactionNotActiveError is TransactionNotActiveError
