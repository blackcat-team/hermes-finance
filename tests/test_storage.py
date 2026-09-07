"""Tests for the SQLite schema & migration foundation (stage C1).

All tests are deterministic: no network, no wall-clock time, no
randomness, no environment access. Filesystem mutation happens only
inside pytest-owned temporary directories (``tmp_path``) or in in-memory
SQLite databases (``:memory:``). No repository or runtime files are
touched.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest

import hermes_finance.storage as storage_module
from hermes_finance import FinanceConfig, open_database
from hermes_finance.storage import SCHEMA_VERSION, DatabaseMigrationError

STORAGE_SOURCE: str = Path(storage_module.__file__).read_text(encoding="utf-8")

FIXED_TZ = UTC


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


def insert_transaction(
    conn: sqlite3.Connection,
    *,
    chat_id: int = -100200,
    message_thread_id: int = 7,
    message_id: int = 1,
    update_id: int = 1000,
    direction: str = "income",
    amount_usdt: str = "25",
    category: str = "Groceries",
    source: str = "card",
    comment: str | None = None,
    transaction_date: str = "2026-09-01",
    created_at: str = "2026-09-01T10:00:00+00:00",
    updated_at: str = "2026-09-01T10:00:00+00:00",
    status: str = "active",
    deleted_at: str | None = None,
) -> None:
    """Insert one transaction row with the given field overrides."""
    conn.execute(
        "INSERT INTO transactions ("
        " direction, amount_usdt, category, source, comment,"
        " transaction_date, created_at, updated_at, status, deleted_at,"
        " chat_id, message_thread_id, message_id, update_id"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            direction,
            amount_usdt,
            category,
            source,
            comment,
            transaction_date,
            created_at,
            updated_at,
            status,
            deleted_at,
            chat_id,
            message_thread_id,
            message_id,
            update_id,
        ),
    )


def insert_processed_update(
    conn: sqlite3.Connection,
    *,
    update_id: int = 5000,
    chat_id: int = -100200,
    message_thread_id: int = 7,
    message_id: int = 1,
    processed_at: str = "2026-09-01T10:00:00+00:00",
) -> None:
    """Insert one processed_updates row with the given field overrides."""
    conn.execute(
        "INSERT INTO processed_updates ("
        " update_id, chat_id, message_thread_id, message_id, processed_at"
        ") VALUES (?, ?, ?, ?, ?)",
        (update_id, chat_id, message_thread_id, message_id, processed_at),
    )


def table_names(conn: sqlite3.Connection) -> set[str]:
    """All table names present in the database schema."""
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {str(row[0]) for row in rows}


def user_version(conn: sqlite3.Connection) -> int:
    """The database schema version (PRAGMA user_version)."""
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0])


def column_declarations(conn: sqlite3.Connection, table: str) -> dict[str, str]:
    """Mapping of column name -> declared type from PRAGMA table_info."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row[1]): str(row[2]) for row in rows}


# ---------------------------------------------------------------------------
# open / bootstrap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_config", ["data/finance.sqlite", None, 123, object()])
def test_open_database_rejects_non_config_type(bad_config: Any) -> None:
    with pytest.raises(TypeError):
        open_database(bad_config)


def test_open_database_accepts_finance_config_and_returns_connection() -> None:
    conn = open_memory()
    try:
        assert isinstance(conn, sqlite3.Connection)
        assert user_version(conn) == 1
    finally:
        conn.close()


def test_open_database_enables_foreign_keys() -> None:
    conn = open_memory()
    try:
        row = conn.execute("PRAGMA foreign_keys").fetchone()
        assert int(row[0]) == 1
    finally:
        conn.close()


def test_open_database_schema_version_is_one() -> None:
    conn = open_memory()
    try:
        assert user_version(conn) == SCHEMA_VERSION
        assert SCHEMA_VERSION == 1
    finally:
        conn.close()


def test_open_database_creates_required_tables() -> None:
    conn = open_memory()
    try:
        names = table_names(conn)
        assert "transactions" in names
        assert "processed_updates" in names
    finally:
        conn.close()


def test_open_database_returns_caller_owned_connections() -> None:
    first = open_memory()
    second = open_memory()
    try:
        assert first is not second
    finally:
        first.close()
        second.close()


def test_open_database_missing_parent_directory_fails_without_mkdir(
    tmp_path: Path,
) -> None:
    missing_dir = tmp_path / "does" / "not" / "exist"
    config = file_config(missing_dir / "finance.sqlite")
    with pytest.raises(sqlite3.Error):
        open_database(config)
    assert not missing_dir.exists()


# ---------------------------------------------------------------------------
# idempotent reopen
# ---------------------------------------------------------------------------


def test_reopen_is_idempotent_and_preserves_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "finance.sqlite"

    first = open_database(file_config(db_path))
    insert_transaction(first, message_id=11, update_id=1011)
    insert_transaction(
        first,
        message_id=12,
        update_id=1012,
        amount_usdt="123.456789",
        direction="expense",
    )
    first.commit()
    first.close()

    second = open_database(file_config(db_path))
    try:
        assert user_version(second) == 1
        rows = second.execute(
            "SELECT amount_usdt FROM transactions ORDER BY id"
        ).fetchall()
        assert [str(row[0]) for row in rows] == ["25", "123.456789"]
        # Schema was not duplicated or corrupted by the second open.
        names = table_names(second)
        assert names == {"transactions", "processed_updates"}
    finally:
        second.close()


def test_reopen_does_not_reapply_migration(tmp_path: Path) -> None:
    db_path = tmp_path / "finance.sqlite"

    first = open_database(file_config(db_path))
    first.close()

    second = open_database(file_config(db_path))
    try:
        assert user_version(second) == 1
        assert table_names(second) == {"transactions", "processed_updates"}
    finally:
        second.close()


# ---------------------------------------------------------------------------
# future schema version
# ---------------------------------------------------------------------------


def test_future_schema_version_is_rejected(tmp_path: Path) -> None:
    db_path = tmp_path / "finance.sqlite"
    raw = sqlite3.connect(db_path)
    raw.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    raw.close()

    with pytest.raises(DatabaseMigrationError):
        open_database(file_config(db_path))

    raw = sqlite3.connect(db_path)
    try:
        assert user_version(raw) == SCHEMA_VERSION + 1
    finally:
        raw.close()


def test_future_schema_version_error_is_public_exception() -> None:
    assert issubclass(DatabaseMigrationError, Exception)


# ---------------------------------------------------------------------------
# migration atomicity
# ---------------------------------------------------------------------------


def test_failed_migration_does_not_advance_user_version(tmp_path: Path) -> None:
    db_path = tmp_path / "finance.sqlite"
    # Pre-create a conflicting table so migration v1 fails midway: the
    # transactions DDL succeeds first, then processed_updates collides.
    raw = sqlite3.connect(db_path)
    raw.execute("CREATE TABLE processed_updates (dummy INTEGER)")
    raw.close()

    with pytest.raises(DatabaseMigrationError):
        open_database(file_config(db_path))

    raw = sqlite3.connect(db_path)
    try:
        # The rolled-back migration left neither a version bump nor the
        # transactions table it had already created.
        assert user_version(raw) == 0
        assert "transactions" not in table_names(raw)
    finally:
        raw.close()


# ---------------------------------------------------------------------------
# transactions schema
# ---------------------------------------------------------------------------


def test_transactions_required_columns_exist() -> None:
    conn = open_memory()
    try:
        declared = column_declarations(conn, "transactions")
        required = {
            "id",
            "direction",
            "amount_usdt",
            "category",
            "source",
            "comment",
            "transaction_date",
            "created_at",
            "updated_at",
            "status",
            "deleted_at",
            "chat_id",
            "message_thread_id",
            "message_id",
            "update_id",
        }
        assert required <= set(declared)
    finally:
        conn.close()


def test_transactions_amount_usdt_declared_text() -> None:
    conn = open_memory()
    try:
        declared = column_declarations(conn, "transactions")
        assert declared["amount_usdt"] == "TEXT"
    finally:
        conn.close()


@pytest.mark.parametrize("direction", ["transfer", "", "INCOME", "income "])
def test_transactions_rejects_invalid_direction(direction: str) -> None:
    conn = open_memory()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            insert_transaction(conn, direction=direction, message_id=21, update_id=1021)
    finally:
        conn.close()


@pytest.mark.parametrize("status", ["pending", "", "ACTIVE", "deleted "])
def test_transactions_rejects_invalid_status(status: str) -> None:
    conn = open_memory()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            insert_transaction(conn, status=status, message_id=22, update_id=1022)
    finally:
        conn.close()


def test_transactions_rejects_active_with_deleted_at() -> None:
    conn = open_memory()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            insert_transaction(
                conn,
                status="active",
                deleted_at="2026-09-01T11:00:00+00:00",
                message_id=23,
                update_id=1023,
            )
    finally:
        conn.close()


def test_transactions_rejects_deleted_without_deleted_at() -> None:
    conn = open_memory()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            insert_transaction(
                conn, status="deleted", deleted_at=None, message_id=24, update_id=1024
            )
    finally:
        conn.close()


def test_transactions_accepts_deleted_with_deleted_at() -> None:
    conn = open_memory()
    try:
        insert_transaction(
            conn,
            status="deleted",
            deleted_at="2026-09-01T11:00:00+00:00",
            message_id=25,
            update_id=1025,
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        assert int(count) == 1
    finally:
        conn.close()


@pytest.mark.parametrize(
    "category",
    [
        "",
        " ",
        "\t",
        "\n",
        "\r",
        "\v",
        "\f",
        "\t \n",
        "\x1c",
        "\r\x1f \x1d",
    ],
)
def test_transactions_rejects_empty_category(category: str) -> None:
    conn = open_memory()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            insert_transaction(conn, category=category, message_id=26, update_id=1026)
    finally:
        conn.close()


@pytest.mark.parametrize(
    "source",
    [
        "",
        " ",
        "\t",
        "\n",
        "\r",
        "\v",
        "\f",
        "\t \n",
        "\x1c",
        "\r\x1f \x1d",
    ],
)
def test_transactions_rejects_empty_source(source: str) -> None:
    conn = open_memory()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            insert_transaction(conn, source=source, message_id=27, update_id=1027)
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("category", "source"),
    [
        ("Sample", "card"),
        ("Test Seller", "Test Seller"),
        ("A\tB", "A\tB"),
        (" Sample ", " Test Seller "),
        ("Line\nBreak", "Tab\tValue"),
    ],
)
def test_transactions_accepts_non_blank_text_with_whitespace(
    category: str, source: str
) -> None:
    conn = open_memory()
    try:
        insert_transaction(
            conn, category=category, source=source, message_id=33, update_id=1033
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        assert int(count) == 1
    finally:
        conn.close()


def test_transactions_rejects_zero_chat_id() -> None:
    conn = open_memory()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            insert_transaction(conn, chat_id=0, message_id=28, update_id=1028)
    finally:
        conn.close()


@pytest.mark.parametrize("thread_id", [-7, 0])
def test_transactions_rejects_non_positive_thread_id(thread_id: int) -> None:
    conn = open_memory()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            insert_transaction(
                conn, message_thread_id=thread_id, message_id=29, update_id=1029
            )
    finally:
        conn.close()


@pytest.mark.parametrize("message_id", [-3, 0])
def test_transactions_rejects_non_positive_message_id(message_id: int) -> None:
    conn = open_memory()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            insert_transaction(conn, message_id=message_id, update_id=1030)
    finally:
        conn.close()


@pytest.mark.parametrize("update_id", [-1])
def test_transactions_rejects_negative_update_id(update_id: int) -> None:
    conn = open_memory()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            insert_transaction(conn, message_id=31, update_id=update_id)
    finally:
        conn.close()


def test_transactions_accepts_zero_update_id_and_comment_null() -> None:
    conn = open_memory()
    try:
        insert_transaction(conn, message_id=32, update_id=0, comment=None)
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        assert int(count) == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# logical message idempotency: UNIQUE (chat_id, message_id)
# ---------------------------------------------------------------------------


def test_transactions_rejects_duplicate_chat_and_message() -> None:
    conn = open_memory()
    try:
        insert_transaction(conn, chat_id=-100200, message_id=40, update_id=1040)
        with pytest.raises(sqlite3.IntegrityError):
            insert_transaction(conn, chat_id=-100200, message_id=40, update_id=1041)
    finally:
        conn.close()


def test_transactions_accepts_same_message_id_in_different_chat() -> None:
    conn = open_memory()
    try:
        insert_transaction(conn, chat_id=-100200, message_id=41, update_id=1041)
        insert_transaction(conn, chat_id=-100201, message_id=41, update_id=1042)
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        assert int(count) == 2
    finally:
        conn.close()


def test_transactions_rejects_same_message_with_different_thread() -> None:
    conn = open_memory()
    try:
        insert_transaction(
            conn, chat_id=-100200, message_thread_id=7, message_id=42, update_id=1043
        )
        with pytest.raises(sqlite3.IntegrityError):
            insert_transaction(
                conn,
                chat_id=-100200,
                message_thread_id=8,
                message_id=42,
                update_id=1044,
            )
    finally:
        conn.close()


def test_transactions_rejects_same_message_with_different_update() -> None:
    conn = open_memory()
    try:
        insert_transaction(conn, chat_id=-100200, message_id=43, update_id=1045)
        with pytest.raises(sqlite3.IntegrityError):
            insert_transaction(conn, chat_id=-100200, message_id=43, update_id=1046)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# update idempotency
# ---------------------------------------------------------------------------


def test_transactions_reject_duplicate_update_id() -> None:
    conn = open_memory()
    try:
        insert_transaction(conn, chat_id=-100200, message_id=44, update_id=1047)
        with pytest.raises(sqlite3.IntegrityError):
            insert_transaction(conn, chat_id=-100201, message_id=45, update_id=1047)
    finally:
        conn.close()


def test_processed_updates_reject_duplicate_update_id() -> None:
    conn = open_memory()
    try:
        insert_processed_update(conn, update_id=5001, message_id=46)
        with pytest.raises(sqlite3.IntegrityError):
            insert_processed_update(conn, update_id=5001, message_id=47)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# processed_updates schema
# ---------------------------------------------------------------------------


def test_processed_updates_required_columns_exist() -> None:
    conn = open_memory()
    try:
        declared = column_declarations(conn, "processed_updates")
        required = {
            "update_id",
            "chat_id",
            "message_thread_id",
            "message_id",
            "processed_at",
        }
        assert required <= set(declared)
    finally:
        conn.close()


def test_processed_updates_accepts_valid_row() -> None:
    conn = open_memory()
    try:
        insert_processed_update(conn, update_id=5002, message_id=48)
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM processed_updates").fetchone()[0]
        assert int(count) == 1
    finally:
        conn.close()


def test_processed_updates_rejects_zero_chat_id() -> None:
    conn = open_memory()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            insert_processed_update(conn, update_id=5003, chat_id=0, message_id=49)
    finally:
        conn.close()


@pytest.mark.parametrize("thread_id", [-7, 0])
def test_processed_updates_rejects_non_positive_thread_id(thread_id: int) -> None:
    conn = open_memory()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            insert_processed_update(
                conn, update_id=5004, message_thread_id=thread_id, message_id=50
            )
    finally:
        conn.close()


@pytest.mark.parametrize("message_id", [-3, 0])
def test_processed_updates_rejects_non_positive_message_id(message_id: int) -> None:
    conn = open_memory()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            insert_processed_update(conn, update_id=5005, message_id=message_id)
    finally:
        conn.close()


@pytest.mark.parametrize("update_id", [-1])
def test_processed_updates_rejects_negative_update_id(update_id: int) -> None:
    conn = open_memory()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            insert_processed_update(conn, update_id=update_id, message_id=51)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# decimal text round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("amount", "message_id"),
    [("25", 60), ("0.000001", 61), ("123.456789", 62)],
)
def test_amount_usdt_round_trips_exact_text(amount: str, message_id: int) -> None:
    conn = open_memory()
    try:
        insert_transaction(
            conn, amount_usdt=amount, message_id=message_id, update_id=1000 + message_id
        )
        conn.commit()
        row = conn.execute(
            "SELECT amount_usdt FROM transactions WHERE message_id = ?",
            (message_id,),
        ).fetchone()
        assert isinstance(row[0], str)
        assert row[0] == amount
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# side effects / scope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    [
        "os.environ",
        "getenv",
        "mkdir",
        "makedirs",
        "datetime.now",
        "utcnow",
        "time.time",
        "perf_counter",
        "monotonic",
        "journal_mode",
        "busy_timeout",
    ],
)
def test_storage_source_avoids_forbidden_operations(forbidden: str) -> None:
    assert forbidden not in STORAGE_SOURCE


@pytest.mark.parametrize(
    "crud_pattern",
    ["def insert_transaction", "def get_transaction", "def list_", "def delete_"],
)
def test_storage_introduces_no_repository_crud(crud_pattern: str) -> None:
    assert crud_pattern not in STORAGE_SOURCE


def test_storage_public_api_is_small_and_deliberate() -> None:
    assert storage_module.__all__ == [
        "SCHEMA_VERSION",
        "DatabaseMigrationError",
        "open_database",
    ]
    assert storage_module.SCHEMA_VERSION == 1
    assert callable(storage_module.open_database)
