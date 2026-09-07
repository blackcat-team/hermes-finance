"""SQLite persistence foundation for Hermes Finance (stage C1).

This module provides the smallest deliberate persistence bootstrap layer:

- :data:`SCHEMA_VERSION`: the schema version this codebase supports
- :func:`open_database`: open (and if necessary migrate) the configured
  SQLite database and return a caller-owned, ready-to-use connection
- :class:`DatabaseMigrationError`: the public failure type for refused or
  failed migrations

Schema versioning uses the SQLite-native ``PRAGMA user_version`` mechanism.
Version 1 creates the ``transactions`` table (with Telegram provenance and
DB-level idempotency constraints) and the ``processed_updates`` table.

Design boundaries of this slice:

- stdlib :mod:`sqlite3` only; no ORM, no third-party migration framework
- ``amount_usdt`` is declared TEXT: exact ``Decimal`` representations are
  preserved as decimal text and never converted to binary floating point
- the domain layer remains authoritative; the CHECK constraints below are
  inexpensive guards against obvious corruption, not a re-implementation of
  domain validation
- no repository CRUD, row mapping, or Decimal serialisation is implemented
  here; those belong to the later repository slice
- the module never consults the wall clock, never reads the process
  environment, never creates directories, and never resolves or invents
  database paths: the configured :class:`~hermes_finance.config.FinanceConfig`
  database path is passed to SQLite exactly as supplied
- every call returns a fresh, caller-owned connection; there is no hidden
  global connection or singleton
"""

from __future__ import annotations

import sqlite3
from typing import Final

from hermes_finance.config import FinanceConfig

__all__ = [
    "SCHEMA_VERSION",
    "DatabaseMigrationError",
    "open_database",
]

#: Schema version supported by this codebase. Databases reporting a higher
#: ``PRAGMA user_version`` are refused instead of silently downgraded.
SCHEMA_VERSION: Final[int] = 1


class DatabaseMigrationError(Exception):
    """Raised when a database cannot be migrated to the supported schema.

    Cases covered:

    - the database reports a schema version newer than :data:`SCHEMA_VERSION`
      (future databases are never silently downgraded or operated on)
    - applying a migration failed; the failing migration is rolled back so
      the stored schema version never claims an unapplied schema
    """


# DDL for migration v1. Statements are executed individually inside one
# explicit transaction (executescript is intentionally avoided because it
# commits implicitly and would break migration atomicity).
#
# Blank-text guards: SQLite's bare trim(X) removes only ASCII space U+0020,
# so a whitespace-only value such as "\t" or "\t \n" would pass a bare
# length(trim(X)) > 0 CHECK. The guards below pass an explicit trim
# character set covering the ordinary ASCII whitespace Python treats as
# str.isspace()-blank: U+0009..U+000D, U+001C..U+001F, and U+0020. The
# domain layer remains authoritative for full text normalisation.
_BLANK_TEXT_TRIM_CHARACTERS: Final[
    str
] = "\t\n\x0b\x0c\r\x1c\x1d\x1e\x1f "

_MIGRATION_V1_STATEMENTS: Final[tuple[str, ...]] = (
    f"""
    CREATE TABLE transactions (
        id INTEGER PRIMARY KEY,
        direction TEXT NOT NULL
            CHECK (direction IN ('income', 'expense')),
        amount_usdt TEXT NOT NULL,
        category TEXT NOT NULL
            CHECK (length(trim(category, '{_BLANK_TEXT_TRIM_CHARACTERS}')) > 0),
        source TEXT NOT NULL
            CHECK (length(trim(source, '{_BLANK_TEXT_TRIM_CHARACTERS}')) > 0),
        comment TEXT NULL,
        transaction_date TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        status TEXT NOT NULL
            CHECK (status IN ('active', 'deleted')),
        deleted_at TEXT NULL,
        chat_id INTEGER NOT NULL
            CHECK (chat_id != 0),
        message_thread_id INTEGER NOT NULL
            CHECK (message_thread_id > 0),
        message_id INTEGER NOT NULL
            CHECK (message_id > 0),
        update_id INTEGER NOT NULL
            CHECK (update_id >= 0),
        CHECK (
            (status = 'active' AND deleted_at IS NULL)
            OR (status = 'deleted' AND deleted_at IS NOT NULL)
        ),
        UNIQUE (chat_id, message_id),
        UNIQUE (update_id)
    )
    """,
    """
    CREATE TABLE processed_updates (
        update_id INTEGER PRIMARY KEY
            CHECK (update_id >= 0),
        chat_id INTEGER NOT NULL
            CHECK (chat_id != 0),
        message_thread_id INTEGER NOT NULL
            CHECK (message_thread_id > 0),
        message_id INTEGER NOT NULL
            CHECK (message_id > 0),
        processed_at TEXT NOT NULL
    )
    """,
)


def _read_user_version(connection: sqlite3.Connection) -> int:
    """Read the database schema version via ``PRAGMA user_version``."""
    row = connection.execute("PRAGMA user_version").fetchone()
    version = row[0]
    if not isinstance(version, int):  # pragma: no cover - SQLite always returns int
        raise DatabaseMigrationError(f"unexpected user_version value: {version!r}")
    return version


def _set_user_version(connection: sqlite3.Connection, version: int) -> None:
    """Set the database schema version inside the current transaction.

    ``PRAGMA user_version`` is transactional in SQLite: if the surrounding
    transaction is rolled back, the stored version is rolled back too, so a
    failed migration can never leave the database claiming an unapplied
    schema version.
    """
    connection.execute(f"PRAGMA user_version = {int(version)}")


def _migrate_to_v1(connection: sqlite3.Connection) -> None:
    """Apply migration v1 atomically (empty database -> schema version 1)."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        for statement in _MIGRATION_V1_STATEMENTS:
            connection.execute(statement)
        _set_user_version(connection, SCHEMA_VERSION)
    except BaseException:
        connection.rollback()
        raise
    connection.commit()


def _apply_migrations(connection: sqlite3.Connection) -> None:
    """Bring the database to :data:`SCHEMA_VERSION`, or refuse.

    - version 0 (new or unmigrated): apply migration v1 atomically
    - version 1 (current): no-op, idempotent open
    - version > 1 (future): refuse with :class:`DatabaseMigrationError`
      and leave the database untouched (never silently downgrade)
    - any other lower version: refuse (unknown migration source)
    """
    current = _read_user_version(connection)
    if current == SCHEMA_VERSION:
        return
    if current > SCHEMA_VERSION:
        raise DatabaseMigrationError(
            f"database schema version {current} is newer than supported "
            f"version {SCHEMA_VERSION}; refusing to operate"
        )
    if current != 0:
        raise DatabaseMigrationError(
            f"cannot migrate from unknown schema version {current} to "
            f"version {SCHEMA_VERSION}"
        )
    try:
        _migrate_to_v1(connection)
    except sqlite3.Error as error:
        raise DatabaseMigrationError(
            f"migration to schema version {SCHEMA_VERSION} failed: {error}"
        ) from error


def open_database(config: FinanceConfig) -> sqlite3.Connection:
    """Open the configured SQLite database and return a migrated connection.

    The caller owns the returned :class:`sqlite3.Connection` and is
    responsible for closing it. There is no shared or cached connection.

    Behaviour:

    - ``config`` must be a :class:`~hermes_finance.config.FinanceConfig`;
      anything else is rejected with :class:`TypeError` before any
      attribute access
    - the configured ``database_path`` is handed to SQLite exactly as
      supplied: it is never resolved, no parent directories are created,
      and no default path is invented. SQLite itself may create the
      database file when its parent directory already exists; a missing
      parent directory fails with the underlying SQLite error
    - foreign key enforcement is enabled on the returned connection via
      ``PRAGMA foreign_keys = ON`` (executed outside any transaction so it
      takes effect)
    - required migrations are applied before the connection is returned;
      a database at the supported schema version is a no-op open

    Failure semantics: if opening or migrating fails, the connection is
    closed before the failure propagates; a half-initialized connection is
    never returned. A refused (future schema) or failed migration raises
    :class:`DatabaseMigrationError`; an unopenable path raises the
    underlying :class:`sqlite3.Error`.

    No WAL/busy-timeout/backup tuning is configured here; those belong to
    later operations work.
    """
    if not isinstance(config, FinanceConfig):
        raise TypeError(
            f"config must be a FinanceConfig, got {type(config).__name__!r}"
        )

    connection = sqlite3.connect(config.database_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        _apply_migrations(connection)
    except BaseException:
        connection.close()
        raise
    return connection
