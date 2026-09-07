"""Tests for the idempotent ingest service and its C3 repository primitives.

All tests are deterministic: no network, no wall-clock time, no
randomness, no environment access, no threads or processes.
Race-shaped semantics are exercised deterministically by monkeypatching
only at the service/repository call boundary to simulate another writer
winning after an earlier read; the core SQLite atomicity guarantees
themselves remain covered by the accepted stage-C2 tests.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import hermes_finance
import hermes_finance.ingest as ingest_module
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
from hermes_finance.ingest import (
    IngestConsistencyError,
    IngestDisposition,
    IngestResult,
    ingest_transaction,
)
from hermes_finance.repository import (
    RepositoryDataError,
    RepositoryTransactionError,
    find_transaction_by_message,
    find_transaction_by_update,
    get_processed_update_ref,
    record_processed_update,
)

INGEST_SOURCE: str = Path(ingest_module.__file__).read_text(encoding="utf-8")
REPOSITORY_SOURCE: str = Path(repository_module.__file__).read_text(encoding="utf-8")

FIXED_TZ = UTC
NON_UTC_TZ = timezone(timedelta(hours=5, minutes=30))

FIXED_DATE = date(2026, 9, 1)
FIXED_CREATED_AT = datetime(2026, 9, 1, 10, 0, 0, tzinfo=FIXED_TZ)
FIXED_UPDATED_AT = datetime(2026, 9, 1, 11, 30, 0, tzinfo=FIXED_TZ)
FIXED_PROCESSED_AT = datetime(2026, 9, 1, 12, 15, 0, tzinfo=FIXED_TZ)


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
    chat_id: int = -100,
    message_thread_id: int = 7,
    message_id: int = 50,
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


def ingest_fixed(
    conn: sqlite3.Connection,
    *,
    transaction: Transaction | None = None,
    ref: TelegramMessageRef | None = None,
    processed_at: datetime | None = None,
) -> IngestResult:
    """ingest_transaction with the fixed test values and overrides."""
    return ingest_transaction(
        conn,
        transaction if transaction is not None else make_transaction(),
        ref if ref is not None else make_ref(),
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


def processed_update_row(conn: sqlite3.Connection, update_id: int) -> Any:
    """The full processed_updates row for the given update_id, or None."""
    return conn.execute(
        "SELECT * FROM processed_updates WHERE update_id = ?", (update_id,)
    ).fetchone()


def insert_raw_processed_update(
    conn: sqlite3.Connection,
    *,
    update_id: int = 1000,
    chat_id: Any = -100,
    message_thread_id: Any = 7,
    message_id: Any = 50,
    processed_at: str = "2026-09-01T09:00:00+00:00",
) -> None:
    """Insert one raw processed_updates row (bypassing the repository)."""
    conn.execute(
        "INSERT INTO processed_updates ("
        " update_id, chat_id, message_thread_id, message_id, processed_at"
        ") VALUES (?, ?, ?, ?, ?)",
        (update_id, chat_id, message_thread_id, message_id, processed_at),
    )
    conn.commit()


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
        "chat_id": -100,
        "message_thread_id": 7,
        "message_id": 50,
        "update_id": 1000,
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
# ingest input contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_connection", ["not-a-connection", None, 5, object()])
def test_ingest_rejects_non_connection(bad_connection: Any) -> None:
    with pytest.raises(TypeError):
        ingest_fixed(bad_connection)


@pytest.mark.parametrize("bad_transaction", [None, "transaction", 5, object()])
def test_ingest_rejects_non_transaction(bad_transaction: Any) -> None:
    conn = open_memory()
    try:
        with pytest.raises(TypeError):
            ingest_transaction(
                conn, bad_transaction, make_ref(), processed_at=FIXED_PROCESSED_AT
            )
    finally:
        conn.close()


@pytest.mark.parametrize("bad_provenance", [None, "ref", 5, object()])
def test_ingest_rejects_non_provenance(bad_provenance: Any) -> None:
    conn = open_memory()
    try:
        with pytest.raises(TypeError):
            ingest_transaction(
                conn, make_transaction(), bad_provenance, processed_at=FIXED_PROCESSED_AT
            )
    finally:
        conn.close()


def test_ingest_rejects_already_persisted_transaction() -> None:
    conn = open_memory()
    try:
        result = ingest_fixed(conn)
        persisted = result.transaction
        assert persisted is not None
        with pytest.raises(RepositoryDataError):
            ingest_fixed(conn, transaction=persisted, ref=make_ref(message_id=51))
        assert count_transactions(conn) == 1
        assert count_processed_updates(conn) == 1
    finally:
        conn.close()


def test_ingest_rejects_naive_processed_at() -> None:
    conn = open_memory()
    try:
        naive = datetime(2026, 9, 1, 12, 15, 0)  # noqa: DTZ001 - naive is the point
        with pytest.raises(ValueError):
            ingest_fixed(conn, processed_at=naive)
        assert count_transactions(conn) == 0
        assert count_processed_updates(conn) == 0
    finally:
        conn.close()


def test_ingest_does_not_mutate_inputs() -> None:
    conn = open_memory()
    try:
        original = make_transaction()
        ref = make_ref()
        ingest_fixed(conn, transaction=original, ref=ref)
        assert original.transaction_id is None
        assert original == make_transaction()
        assert ref == make_ref()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CREATED
# ---------------------------------------------------------------------------


def test_created_disposition_and_both_rows_exist() -> None:
    conn = open_memory()
    try:
        result = ingest_fixed(conn)
        assert isinstance(result, IngestResult)
        assert result.disposition is IngestDisposition.CREATED
        assert count_transactions(conn) == 1
        assert count_processed_updates(conn) == 1
        assert conn.in_transaction is False
    finally:
        conn.close()


def test_created_returns_persisted_transaction_and_original_stays_unpersisted() -> None:
    conn = open_memory()
    try:
        original = make_transaction()
        result = ingest_fixed(conn, transaction=original)
        assert result.transaction.transaction_id == "1"
        assert result.transaction == make_transaction(transaction_id="1")
        assert original.transaction_id is None
        assert original == make_transaction()
    finally:
        conn.close()


def test_created_processed_provenance_is_exact() -> None:
    conn = open_memory()
    try:
        ingest_fixed(conn)
        row = processed_update_row(conn, update_id=1000)
        assert row is not None
        assert row[0] == 1000
        assert row[1] == -100
        assert row[2] == 7
        assert row[3] == 50
        assert row[4] == FIXED_PROCESSED_AT.isoformat()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# DUPLICATE_MESSAGE
# ---------------------------------------------------------------------------


def test_same_message_new_update_is_duplicate_message() -> None:
    conn = open_memory()
    try:
        first = ingest_fixed(conn)
        result = ingest_fixed(conn, ref=make_ref(update_id=1001))

        assert result.disposition is IngestDisposition.DUPLICATE_MESSAGE
        assert result.transaction == first.transaction
        assert count_transactions(conn) == 1
        assert count_processed_updates(conn) == 2

        row = processed_update_row(conn, update_id=1001)
        assert row is not None
        assert (row[0], row[1], row[2], row[3]) == (1001, -100, 7, 50)
        assert row[4] == FIXED_PROCESSED_AT.isoformat()

        stored = find_transaction_by_message(
            conn, TelegramMessageIdentity(chat_id=-100, message_id=50)
        )
        assert stored == first.transaction
    finally:
        conn.close()


def test_duplicate_message_existing_transaction_wins_over_business_fields() -> None:
    conn = open_memory()
    try:
        first = ingest_fixed(conn)
        edited = make_transaction(
            direction=Direction.INCOME,
            amount_usdt=Decimal(999),
            category="Other",
            source="cash",
            comment="changed",
        )
        result = ingest_fixed(conn, transaction=edited, ref=make_ref(update_id=1001))

        assert result.disposition is IngestDisposition.DUPLICATE_MESSAGE
        assert result.transaction == first.transaction
        stored = find_transaction_by_message(
            conn, TelegramMessageIdentity(chat_id=-100, message_id=50)
        )
        assert stored == first.transaction
        assert stored is not None
        assert stored.amount_usdt == Decimal(25)
        assert stored.category == "Groceries"
        assert count_transactions(conn) == 1
    finally:
        conn.close()


def test_duplicate_message_with_different_thread() -> None:
    conn = open_memory()
    try:
        first = ingest_fixed(conn)
        result = ingest_fixed(
            conn, ref=make_ref(message_thread_id=9, update_id=1001)
        )

        assert result.disposition is IngestDisposition.DUPLICATE_MESSAGE
        assert result.transaction == first.transaction
        assert count_transactions(conn) == 1
        assert count_processed_updates(conn) == 2

        row = processed_update_row(conn, update_id=1001)
        assert row is not None
        assert row[2] == 9
        assert (row[1], row[3]) == (-100, 50)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# DUPLICATE_UPDATE
# ---------------------------------------------------------------------------


def test_replay_original_update_is_duplicate_update_without_writes() -> None:
    conn = open_memory()
    try:
        first = ingest_fixed(conn)
        before_transactions = conn.execute(
            "SELECT * FROM transactions ORDER BY id"
        ).fetchall()
        before_updates = conn.execute(
            "SELECT * FROM processed_updates ORDER BY update_id"
        ).fetchall()

        result = ingest_fixed(conn)

        assert result.disposition is IngestDisposition.DUPLICATE_UPDATE
        assert result.transaction == first.transaction
        assert conn.in_transaction is False
        assert conn.execute("SELECT * FROM transactions ORDER BY id").fetchall() == (
            before_transactions
        )
        assert conn.execute(
            "SELECT * FROM processed_updates ORDER BY update_id"
        ).fetchall() == before_updates
    finally:
        conn.close()


def test_replay_duplicate_message_update_returns_original_transaction() -> None:
    conn = open_memory()
    try:
        first = ingest_fixed(conn)
        duplicate = ingest_fixed(conn, ref=make_ref(update_id=1001))
        assert duplicate.disposition is IngestDisposition.DUPLICATE_MESSAGE

        # A DUPLICATE_MESSAGE update has no transactions row of its own.
        assert (
            find_transaction_by_update(conn, TelegramUpdateIdentity(update_id=1001))
            is None
        )

        result = ingest_fixed(conn, ref=make_ref(update_id=1001))

        assert result.disposition is IngestDisposition.DUPLICATE_UPDATE
        assert result.transaction == first.transaction
        assert result.transaction.transaction_id == first.transaction.transaction_id
        assert count_transactions(conn) == 1
        assert count_processed_updates(conn) == 2
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# conflict / inconsistency
# ---------------------------------------------------------------------------


def test_processed_update_conflicting_chat_and_message() -> None:
    conn = open_memory()
    try:
        ingest_fixed(conn)
        with pytest.raises(IngestConsistencyError):
            ingest_fixed(conn, ref=make_ref(chat_id=-20, message_id=8, update_id=1000))
        assert count_transactions(conn) == 1
        assert count_processed_updates(conn) == 1
        assert (
            find_transaction_by_message(
                conn, TelegramMessageIdentity(chat_id=-20, message_id=8)
            )
            is None
        )
    finally:
        conn.close()


def test_processed_update_conflicting_thread() -> None:
    conn = open_memory()
    try:
        ingest_fixed(conn)
        with pytest.raises(IngestConsistencyError):
            ingest_fixed(conn, ref=make_ref(message_thread_id=9, update_id=1000))
        assert count_transactions(conn) == 1
        assert count_processed_updates(conn) == 1
        row = processed_update_row(conn, update_id=1000)
        assert row is not None
        assert row[2] == 7
    finally:
        conn.close()


def test_orphan_processed_update_raises() -> None:
    conn = open_memory()
    try:
        insert_raw_processed_update(
            conn, update_id=3000, chat_id=-100, message_thread_id=7, message_id=77
        )
        with pytest.raises(IngestConsistencyError):
            ingest_fixed(conn, ref=make_ref(message_id=77, update_id=3000))
        assert count_transactions(conn) == 0
        assert count_processed_updates(conn) == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# unrelated integrity error
# ---------------------------------------------------------------------------


def test_unrelated_integrity_error_propagates() -> None:
    conn = open_memory()
    try:
        # A raw transactions row occupies update_id 2000 for a different
        # logical message and has no processed_updates row, so neither a
        # processed update for 2000 nor a transaction for the incoming
        # logical message exists when the persist attempt fails.
        insert_raw_transaction(conn, chat_id=-999, message_id=999, update_id=2000)

        with pytest.raises(sqlite3.IntegrityError):
            ingest_fixed(conn, ref=make_ref(update_id=2000))

        assert conn.in_transaction is False
        assert count_transactions(conn) == 1
        assert count_processed_updates(conn) == 0
        assert (
            find_transaction_by_message(
                conn, TelegramMessageIdentity(chat_id=-100, message_id=50)
            )
            is None
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# race-shaped semantics (deterministic, call-boundary monkeypatching only)
# ---------------------------------------------------------------------------


def _patch_first_read_to_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Make the service's first processed-update read miss.

    Simulates another writer having recorded the update before the
    ingest attempt while the service's own fast-path read does not see
    it yet. Every later read delegates to the real repository function.
    """
    real_get = get_processed_update_ref
    seen: list[bool] = []

    def first_read_misses(
        connection: sqlite3.Connection, identity: TelegramUpdateIdentity
    ) -> TelegramMessageRef | None:
        if not seen:
            seen.append(True)
            return None
        return real_get(connection, identity)

    monkeypatch.setattr(ingest_module, "get_processed_update_ref", first_read_misses)


def test_persist_conflict_with_now_visible_processed_update_is_duplicate_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = open_memory()
    try:
        first = ingest_fixed(conn)
        insert_raw_processed_update(
            conn, update_id=2000, chat_id=-100, message_thread_id=7, message_id=50
        )
        _patch_first_read_to_miss(monkeypatch)

        result = ingest_fixed(conn, ref=make_ref(update_id=2000))

        assert result.disposition is IngestDisposition.DUPLICATE_UPDATE
        assert result.transaction == first.transaction
        assert count_transactions(conn) == 1
        assert count_processed_updates(conn) == 2
    finally:
        conn.close()


def test_persist_conflict_with_inconsistent_processed_update_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = open_memory()
    try:
        ingest_fixed(conn)
        insert_raw_processed_update(
            conn, update_id=2000, chat_id=-20, message_thread_id=7, message_id=8
        )
        _patch_first_read_to_miss(monkeypatch)

        with pytest.raises(IngestConsistencyError) as excinfo:
            ingest_fixed(conn, ref=make_ref(update_id=2000))

        assert isinstance(excinfo.value.__cause__, sqlite3.IntegrityError)
        assert count_transactions(conn) == 1
        assert count_processed_updates(conn) == 2
    finally:
        conn.close()


def test_persist_conflict_with_orphan_processed_update_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = open_memory()
    try:
        insert_raw_processed_update(
            conn, update_id=2000, chat_id=-100, message_thread_id=7, message_id=50
        )
        _patch_first_read_to_miss(monkeypatch)

        with pytest.raises(IngestConsistencyError) as excinfo:
            ingest_fixed(conn, ref=make_ref(update_id=2000))

        assert isinstance(excinfo.value.__cause__, sqlite3.IntegrityError)
        assert count_transactions(conn) == 0
        assert count_processed_updates(conn) == 1
    finally:
        conn.close()


def test_record_processed_update_race_lost_is_duplicate_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = open_memory()
    try:
        first = ingest_fixed(conn)
        real_record = record_processed_update

        def losing_record(
            connection: sqlite3.Connection,
            provenance: TelegramMessageRef,
            *,
            processed_at: datetime,
        ) -> None:
            # Another writer wins the race and records this exact
            # delivery; our own insert then loses the unique race.
            real_record(connection, provenance, processed_at=processed_at)
            raise sqlite3.IntegrityError(
                "UNIQUE constraint failed: processed_updates.update_id"
            )

        monkeypatch.setattr(ingest_module, "record_processed_update", losing_record)

        result = ingest_fixed(conn, ref=make_ref(update_id=2000))

        assert result.disposition is IngestDisposition.DUPLICATE_UPDATE
        assert result.transaction == first.transaction
        assert count_transactions(conn) == 1
        # The winning writer's processed_updates row for update 2000.
        assert count_processed_updates(conn) == 2
        assert processed_update_row(conn, update_id=2000) is not None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# record_processed_update
# ---------------------------------------------------------------------------


def test_record_processed_update_inserts_exact_row() -> None:
    conn = open_memory()
    try:
        record_processed_update(conn, make_ref(), processed_at=FIXED_PROCESSED_AT)
        row = processed_update_row(conn, update_id=1000)
        assert row is not None
        assert row == (1000, -100, 7, 50, "2026-09-01T12:15:00+00:00")
        assert count_transactions(conn) == 0
        assert conn.in_transaction is False
    finally:
        conn.close()


def test_record_processed_update_non_utc_processed_at_stored_exactly() -> None:
    conn = open_memory()
    try:
        processed_at = datetime(2026, 9, 1, 17, 45, 0, tzinfo=NON_UTC_TZ)
        record_processed_update(conn, make_ref(), processed_at=processed_at)
        row = processed_update_row(conn, update_id=1000)
        assert row is not None
        assert row[4] == processed_at.isoformat()
        assert row[4].endswith("+05:30")
    finally:
        conn.close()


def test_record_processed_update_commit_survives_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "finance.sqlite"
    conn = open_database(file_config(db_path))
    try:
        record_processed_update(
            conn, make_ref(message_id=60, update_id=1060), processed_at=FIXED_PROCESSED_AT
        )
    finally:
        conn.close()

    reopened = open_database(file_config(db_path))
    try:
        assert count_processed_updates(reopened) == 1
        assert count_transactions(reopened) == 0
        row = processed_update_row(reopened, update_id=1060)
        assert row is not None
        assert (row[1], row[2], row[3]) == (-100, 7, 60)
        assert row[4] == FIXED_PROCESSED_AT.isoformat()
    finally:
        reopened.close()


def test_record_processed_update_duplicate_update_id_raises() -> None:
    conn = open_memory()
    try:
        record_processed_update(conn, make_ref(), processed_at=FIXED_PROCESSED_AT)
        with pytest.raises(sqlite3.IntegrityError):
            record_processed_update(
                conn, make_ref(message_id=51), processed_at=FIXED_PROCESSED_AT
            )
        assert conn.in_transaction is False
        assert count_processed_updates(conn) == 1
        assert count_transactions(conn) == 0
    finally:
        conn.close()


def test_record_processed_update_rejects_naive_processed_at() -> None:
    conn = open_memory()
    try:
        naive = datetime(2026, 9, 1, 12, 15, 0)  # noqa: DTZ001 - naive is the point
        with pytest.raises(ValueError):
            record_processed_update(conn, make_ref(), processed_at=naive)
        assert count_processed_updates(conn) == 0
    finally:
        conn.close()


@pytest.mark.parametrize("bad_provenance", [None, "ref", 5, object()])
def test_record_processed_update_rejects_wrong_provenance_type(bad_provenance: Any) -> None:
    conn = open_memory()
    try:
        with pytest.raises(TypeError):
            record_processed_update(
                conn, bad_provenance, processed_at=FIXED_PROCESSED_AT
            )
    finally:
        conn.close()


@pytest.mark.parametrize("bad_connection", ["not-a-connection", None, 5, object()])
def test_record_processed_update_rejects_non_connection(bad_connection: Any) -> None:
    with pytest.raises(TypeError):
        record_processed_update(bad_connection, make_ref(), processed_at=FIXED_PROCESSED_AT)


def test_record_processed_update_rejects_active_caller_transaction() -> None:
    conn = open_memory()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO processed_updates ("
            " update_id, chat_id, message_thread_id, message_id, processed_at"
            ") VALUES (?, ?, ?, ?, ?)",
            (1008, -100, 7, 58, "2026-09-01T09:00:00+00:00"),
        )
        assert conn.in_transaction is True

        with pytest.raises(RepositoryTransactionError):
            record_processed_update(conn, make_ref(), processed_at=FIXED_PROCESSED_AT)

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


class _CommitFailsConnection(sqlite3.Connection):
    """A real sqlite3.Connection whose commit() fails deterministically.

    rollback() keeps its normal behaviour, so this subclass proves that
    a commit failure inside record_processed_update's write transaction
    triggers the rollback path and leaves no pending repository
    transaction. Test-only; the production code has no knowledge of it.
    """

    def commit(self) -> None:
        raise sqlite3.OperationalError("deterministic commit failure")


def test_record_processed_update_commit_failure_triggers_rollback(tmp_path: Path) -> None:
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
        assert conn.in_transaction is False

        with pytest.raises(sqlite3.OperationalError, match="deterministic commit failure"):
            record_processed_update(conn, make_ref(), processed_at=FIXED_PROCESSED_AT)

        # The repository rolled its own transaction back: nothing is
        # pending and the inserted row is not visible.
        assert conn.in_transaction is False
        assert count_processed_updates(conn) == 0
        assert processed_update_row(conn, update_id=1000) is None

        # The connection remains usable after the failure.
        assert conn.execute("SELECT COUNT(*) FROM processed_updates").fetchone()[0] == 0
    finally:
        conn.close()

    # The on-disk database also contains no row.
    verify = sqlite3.connect(db_path)
    try:
        assert int(verify.execute("SELECT COUNT(*) FROM processed_updates").fetchone()[0]) == 0
    finally:
        verify.close()


# ---------------------------------------------------------------------------
# get_processed_update_ref
# ---------------------------------------------------------------------------


def test_get_processed_update_ref_returns_exact_ref() -> None:
    conn = open_memory()
    try:
        ingest_fixed(conn)
        ref = get_processed_update_ref(conn, TelegramUpdateIdentity(update_id=1000))
        assert ref == make_ref()
    finally:
        conn.close()


def test_get_processed_update_ref_missing_returns_none() -> None:
    conn = open_memory()
    try:
        ingest_fixed(conn)
        assert get_processed_update_ref(conn, TelegramUpdateIdentity(update_id=9999)) is None
    finally:
        conn.close()


@pytest.mark.parametrize(
    "bad_identity",
    [
        None,
        5,
        "identity",
        TelegramMessageIdentity(chat_id=-100, message_id=50),
        make_ref(),
    ],
)
def test_get_processed_update_ref_rejects_wrong_identity_type(bad_identity: Any) -> None:
    conn = open_memory()
    try:
        with pytest.raises(TypeError):
            get_processed_update_ref(conn, bad_identity)
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chat_id", "abc"),
        ("message_thread_id", b"\x00\x01"),
        ("message_id", "x"),
    ],
)
def test_get_processed_update_ref_corrupt_row_raises(field: str, value: Any) -> None:
    conn = open_memory()
    try:
        overrides: dict[str, Any] = {field: value}
        insert_raw_processed_update(conn, **overrides)
        with pytest.raises(RepositoryDataError) as excinfo:
            get_processed_update_ref(conn, TelegramUpdateIdentity(update_id=1000))
        assert excinfo.value.__cause__ is not None
    finally:
        conn.close()


def test_get_processed_update_ref_read_is_side_effect_free() -> None:
    conn = open_memory()
    try:
        ingest_fixed(conn)
        before_updates = conn.execute(
            "SELECT * FROM processed_updates ORDER BY update_id"
        ).fetchall()
        before_transactions = conn.execute(
            "SELECT * FROM transactions ORDER BY id"
        ).fetchall()
        assert conn.in_transaction is False

        get_processed_update_ref(conn, TelegramUpdateIdentity(update_id=1000))
        get_processed_update_ref(conn, TelegramUpdateIdentity(update_id=9999))

        assert conn.in_transaction is False
        assert conn.execute(
            "SELECT * FROM processed_updates ORDER BY update_id"
        ).fetchall() == before_updates
        assert conn.execute("SELECT * FROM transactions ORDER BY id").fetchall() == (
            before_transactions
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# exception boundaries
# ---------------------------------------------------------------------------


def test_ingest_active_caller_transaction_propagates_repository_transaction_error() -> None:
    conn = open_memory()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO processed_updates ("
            " update_id, chat_id, message_thread_id, message_id, processed_at"
            ") VALUES (?, ?, ?, ?, ?)",
            (1008, -100, 7, 58, "2026-09-01T09:00:00+00:00"),
        )
        assert conn.in_transaction is True

        with pytest.raises(RepositoryTransactionError):
            ingest_fixed(conn, ref=make_ref(update_id=1009))

        assert conn.in_transaction is True
        conn.rollback()
        assert count_processed_updates(conn) == 0
        assert count_transactions(conn) == 0
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
        "time.time",
        "date.today",
        "today()",
        "perf_counter",
        "monotonic",
        "sqlite3.connect",
        "PRAGMA",
        "CREATE TABLE",
        "import os",
        "import time",
    ],
)
def test_ingest_source_avoids_forbidden_operations(forbidden: str) -> None:
    assert forbidden not in INGEST_SOURCE


@pytest.mark.parametrize(
    "forbidden",
    [
        "datetime.now",
        "utcnow",
        "time.time",
        "date.today",
        "today()",
        "perf_counter",
        "monotonic",
    ],
)
def test_c3_repository_primitives_avoid_wall_clock(forbidden: str) -> None:
    assert forbidden not in REPOSITORY_SOURCE


def test_ingest_public_api_is_small_and_deliberate() -> None:
    assert ingest_module.__all__ == [
        "IngestConsistencyError",
        "IngestDisposition",
        "IngestResult",
        "ingest_transaction",
    ]
    for name in ingest_module.__all__:
        assert callable(getattr(ingest_module, name))


def test_package_re_exports_c3_api() -> None:
    assert hermes_finance.ingest_transaction is ingest_transaction
    assert hermes_finance.IngestDisposition is IngestDisposition
    assert hermes_finance.IngestResult is IngestResult
    assert hermes_finance.IngestConsistencyError is IngestConsistencyError
    assert hermes_finance.record_processed_update is record_processed_update
    assert hermes_finance.get_processed_update_ref is get_processed_update_ref
