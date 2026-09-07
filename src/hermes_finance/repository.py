"""SQLite repository layer for Hermes Finance (stages C2/C3).

This module maps the accepted domain and provenance objects onto the
schema-v1 tables created by :func:`hermes_finance.storage.open_database`:

- :func:`persist_transaction` atomically writes one ``transactions`` row
  and one ``processed_updates`` row inside a single explicit
  ``BEGIN IMMEDIATE`` transaction: either both rows commit or neither
  does, and the SQLite-assigned row id becomes the domain
  ``transaction_id``
- :func:`get_transaction`, :func:`find_transaction_by_message`,
  :func:`find_transaction_by_update`, and :func:`is_update_processed`
  are deterministic, side-effect-free reads
- :func:`record_processed_update` and :func:`get_processed_update_ref`
  are the two bounded stage-C3 primitives backing the idempotent ingest
  service: recording one processed update (and nothing else) under the
  same explicit ``BEGIN IMMEDIATE`` ownership rule, and reconstructing
  the stored provenance of an already processed update; they are
  deliberately not part of ``__all__`` because the accepted C2
  API-surface contract pins that list exactly -- they are re-exported
  by the package ``__init__`` instead

Design boundaries of this slice:

- every public function requires a caller-owned, already migrated
  :class:`sqlite3.Connection`; the module never opens a database
  itself, never creates shared connections, and never accepts
  configuration objects
- ``persist_transaction`` refuses connections that already carry an
  active transaction, so unrelated caller work is never silently
  committed or rolled back
- ``amount_usdt`` is serialised as exact decimal text via
  ``str(Decimal)``: ``Decimal`` -> ``TEXT`` -> ``Decimal`` preserves
  both the value and its decimal exponent, and no binary floating
  point is ever involved
- dates use ``date.isoformat``; timestamps use ``datetime.isoformat``
  with their original UTC offsets and are never converted to local
  time
- ``processed_at`` is an explicit caller-supplied timezone-aware
  timestamp; the module never consults the wall clock
- uniqueness relies entirely on the schema-v1 constraints; a constraint
  conflict surfaces as :class:`sqlite3.IntegrityError` after a full
  rollback, and the high-level idempotent ingest interpretation is
  owned by the stage-C3 ingest service
  (:mod:`hermes_finance.ingest`)
- the bounded stage-D1 read primitives
  :func:`list_active_transactions_recent`,
  :func:`list_active_transactions_by_date`, and
  :func:`list_active_transactions_in_month` return only
  ``TransactionStatus.ACTIVE`` rows in deterministic newest-first
  order; they are deliberately not part of ``__all__`` because the
  accepted C2 API-surface contract pins that list exactly -- they are
  consumed by the stage-D1 operations layer
  (:mod:`hermes_finance.operations`) instead
- the bounded stage-D2 mutation primitives
  :func:`replace_active_transaction_fields` and
  :func:`mark_transaction_deleted` atomically replace the business
  fields of one ACTIVE transaction and atomically mark one transaction
  DELETED, both inside one explicit ``BEGIN IMMEDIATE`` transaction
  under the same ownership rule; they are also deliberately not part
  of ``__all__`` -- the stage-D2 mutation service
  (:mod:`hermes_finance.mutations`) owns their public error
  interpretation
- no restore, hard-delete, provenance mutation, or reporting queries
  are implemented here
"""

from __future__ import annotations

import re
import sqlite3
from calendar import monthrange
from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Final

from hermes_finance.domain import (
    Direction,
    Transaction,
    TransactionStatus,
    normalize_usdt_amount,
    require_aware_datetime,
    require_calendar_date,
)
from hermes_finance.provenance import (
    TelegramMessageIdentity,
    TelegramMessageRef,
    TelegramUpdateIdentity,
)

__all__ = [
    "RepositoryDataError",
    "RepositoryTransactionError",
    "find_transaction_by_message",
    "find_transaction_by_update",
    "get_transaction",
    "is_update_processed",
    "persist_transaction",
]


class RepositoryDataError(Exception):
    """Raised when data handed to or read by the repository is invalid.

    Covers two deterministic cases:

    - a domain object handed to :func:`persist_transaction` violates
      the repository input contract (for example a transaction that is
      already persisted)
    - a persisted row cannot be mapped back into a valid
      :class:`~hermes_finance.domain.Transaction` because the stored
      data is corrupt; the original exception is always chained as the
      cause
    """


class RepositoryTransactionError(Exception):
    """Raised when the explicit write transaction cannot be owned.

    Covers the case where the caller hands over a connection that
    already carries an active transaction: the repository refuses to
    nest inside it and never commits or rolls back caller-owned work.
    """


#: Repository transaction IDs are positive decimal strings produced by
#: ``str(rowid)``: no sign, no leading zero, no fractional part.
_TRANSACTION_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[1-9][0-9]*")

_INSERT_TRANSACTION_SQL: Final[str] = (
    "INSERT INTO transactions ("
    " direction, amount_usdt, category, source, comment,"
    " transaction_date, created_at, updated_at, status, deleted_at,"
    " chat_id, message_thread_id, message_id, update_id"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

_INSERT_PROCESSED_UPDATE_SQL: Final[str] = (
    "INSERT INTO processed_updates ("
    " update_id, chat_id, message_thread_id, message_id, processed_at"
    ") VALUES (?, ?, ?, ?, ?)"
)

_SELECT_TRANSACTION_BY_ID: Final[str] = (
    "SELECT id, direction, amount_usdt, category, source, comment,"
    " transaction_date, created_at, updated_at, status, deleted_at"
    " FROM transactions WHERE id = ?"
)

_SELECT_TRANSACTION_BY_MESSAGE: Final[str] = (
    "SELECT id, direction, amount_usdt, category, source, comment,"
    " transaction_date, created_at, updated_at, status, deleted_at"
    " FROM transactions WHERE chat_id = ? AND message_id = ?"
)

_SELECT_TRANSACTION_BY_UPDATE: Final[str] = (
    "SELECT id, direction, amount_usdt, category, source, comment,"
    " transaction_date, created_at, updated_at, status, deleted_at"
    " FROM transactions WHERE update_id = ?"
)

_SELECT_PROCESSED_UPDATE: Final[str] = (
    "SELECT 1 FROM processed_updates WHERE update_id = ?"
)

_SELECT_PROCESSED_UPDATE_REF: Final[str] = (
    "SELECT update_id, chat_id, message_thread_id, message_id"
    " FROM processed_updates WHERE update_id = ?"
)

# Stage-D1 bounded read queries. The column list and row mapper are exactly
# the accepted C2 ones: there is no second Transaction reconstruction path.
# Ordering is always ``transaction_date DESC, id DESC`` (business date
# first, deterministic SQLite insertion order within a date) and visibility
# is always ACTIVE-only.
_SELECT_ACTIVE_ORDERED_BASE: Final[str] = (
    "SELECT id, direction, amount_usdt, category, source, comment,"
    " transaction_date, created_at, updated_at, status, deleted_at"
    " FROM transactions WHERE status = ?"
)
_ORDER_DESC: Final[str] = " ORDER BY transaction_date DESC, id DESC"

_SELECT_ACTIVE_RECENT: Final[str] = _SELECT_ACTIVE_ORDERED_BASE + _ORDER_DESC + " LIMIT ?"

_SELECT_ACTIVE_BY_DATE: Final[str] = (
    _SELECT_ACTIVE_ORDERED_BASE + " AND transaction_date = ?" + _ORDER_DESC
)

_SELECT_ACTIVE_IN_MONTH: Final[str] = (
    _SELECT_ACTIVE_ORDERED_BASE
    + " AND transaction_date >= ? AND transaction_date <= ?"
    + _ORDER_DESC
)

#: Smallest and largest calendar year representable by :class:`datetime.date`.
_MIN_YEAR: Final[int] = 1
_MAX_YEAR: Final[int] = 9999

#: Calendar month bounds.
_MIN_MONTH: Final[int] = 1
_MAX_MONTH: Final[int] = 12


def _require_connection(connection: sqlite3.Connection) -> None:
    """Require a real :class:`sqlite3.Connection` before any use."""
    if not isinstance(connection, sqlite3.Connection):
        raise TypeError(
            f"connection must be a sqlite3.Connection, got {type(connection).__name__!r}"
        )


def _parse_transaction_id(value: object) -> int:
    """Validate a repository transaction ID string and return its ``int``.

    Only positive decimal strings without sign, leading zeros, or
    fractional parts are accepted; anything else is rejected cleanly.
    """
    if not isinstance(value, str):
        raise TypeError(f"transaction_id must be a string, got {type(value).__name__!r}")
    if _TRANSACTION_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"transaction_id is not a valid repository transaction ID: {value!r}")
    return int(value)


def _require_row_text(value: object, column: str) -> str:
    """Require a text value for a NOT NULL text column.

    Raises an ordinary :class:`TypeError` for non-text values (for
    example a SQLite BLOB or INTEGER stored in a TEXT column); the
    single repository error boundary in :func:`_row_to_transaction`
    converts it into a chained :class:`RepositoryDataError`.
    """
    if not isinstance(value, str):
        raise TypeError(f"transactions.{column} must be text, got {value!r}")
    return value


def _optional_row_text(value: object, column: str) -> str | None:
    """Require text or ``None`` for a nullable text column."""
    if value is None:
        return None
    return _require_row_text(value, column)


def _transaction_row_values(
    transaction: Transaction, provenance: TelegramMessageRef
) -> tuple[str | int | None, ...]:
    """Serialise a Transaction and its provenance into row values.

    ``amount_usdt`` is serialised as exact decimal text via ``str``:
    no binary floating point is involved, and the stored text parses
    back into a :class:`~decimal.Decimal` with the same value and
    decimal exponent. Dates and timestamps are serialised with their
    stdlib ISO representations, keeping the original UTC offsets.
    """
    return (
        transaction.direction.value,
        str(transaction.amount_usdt),
        transaction.category,
        transaction.source,
        transaction.comment,
        transaction.transaction_date.isoformat(),
        transaction.created_at.isoformat(),
        transaction.updated_at.isoformat(),
        transaction.status.value,
        transaction.deleted_at.isoformat() if transaction.deleted_at is not None else None,
        provenance.chat_id,
        provenance.message_thread_id,
        provenance.message_id,
        provenance.update_id,
    )


def _row_to_transaction(row: Sequence[object]) -> Transaction:
    """Reconstruct a :class:`Transaction` from a transactions row.

    All decoding and the authoritative domain construction happen inside
    a single error boundary: any ordinary deterministic mapping or
    validation failure (``TypeError`` for wrong stored types such as a
    BLOB in a TEXT column, ``ValueError`` for malformed ISO text or
    invalid enum values, ``ArithmeticError`` for malformed decimal
    text) is converted into a :class:`RepositoryDataError` with the
    original exception preserved as ``__cause__``. Arbitrary
    ``KeyError``/``IndexError``/``AttributeError`` never escape, and
    non-text SQLite values are never silently coerced into strings.
    """
    try:
        (
            raw_id,
            raw_direction,
            raw_amount,
            raw_category,
            raw_source,
            raw_comment,
            raw_transaction_date,
            raw_created_at,
            raw_updated_at,
            raw_status,
            raw_deleted_at,
        ) = row
    except (TypeError, ValueError) as error:
        raise RepositoryDataError(f"unexpected transactions row shape: {row!r}") from error

    try:
        if not isinstance(raw_id, int) or isinstance(raw_id, bool):
            raise TypeError(f"transactions.id must be an integer, got {raw_id!r}")
        return Transaction(
            direction=Direction(_require_row_text(raw_direction, "direction")),
            amount_usdt=Decimal(_require_row_text(raw_amount, "amount_usdt")),
            category=_require_row_text(raw_category, "category"),
            source=_require_row_text(raw_source, "source"),
            comment=_optional_row_text(raw_comment, "comment"),
            transaction_date=date.fromisoformat(
                _require_row_text(raw_transaction_date, "transaction_date")
            ),
            created_at=datetime.fromisoformat(
                _require_row_text(raw_created_at, "created_at")
            ),
            updated_at=datetime.fromisoformat(
                _require_row_text(raw_updated_at, "updated_at")
            ),
            transaction_id=str(raw_id),
            status=TransactionStatus(_require_row_text(raw_status, "status")),
            deleted_at=(
                None
                if raw_deleted_at is None
                else datetime.fromisoformat(_require_row_text(raw_deleted_at, "deleted_at"))
            ),
        )
    except (TypeError, ValueError, ArithmeticError) as error:
        raise RepositoryDataError(
            f"persisted transactions row {raw_id!r} cannot be mapped to a Transaction: {error}"
        ) from error


def persist_transaction(
    connection: sqlite3.Connection,
    transaction: Transaction,
    provenance: TelegramMessageRef,
    *,
    processed_at: datetime,
) -> Transaction:
    """Atomically persist a Transaction and its Telegram provenance.

    Exactly two rows are created inside one explicit ``BEGIN IMMEDIATE``
    transaction: the ``transactions`` row (business fields plus the
    ``chat_id``, ``message_thread_id``, ``message_id``, and ``update_id``
    provenance fields) and the ``processed_updates`` row (the same
    provenance fields plus the caller-supplied ``processed_at``).
    Either both rows commit or neither does: every repository
    operation after ``BEGIN`` -- including the final ``commit`` itself
    -- is inside the rollback-protected region, so any failure after
    ``BEGIN`` triggers a full rollback before the exception propagates
    and never knowingly leaves the repository's own transaction
    pending.

    Contract:

    - ``connection`` must be a caller-owned
      :class:`sqlite3.Connection` prepared by
      :func:`hermes_finance.storage.open_database`; the schema is never
      created, migrated, or tuned here
    - ``connection`` must not already carry an active transaction; the
      repository must own its write transaction so unrelated caller
      work is never silently committed or rolled back
    - ``transaction`` must be an unpersisted
      :class:`~hermes_finance.domain.Transaction`
      (``transaction_id is None``); the input object is never mutated
    - ``provenance`` must be a
      :class:`~hermes_finance.provenance.TelegramMessageRef`
    - ``processed_at`` must be a timezone-aware ``datetime`` supplied
      explicitly by the caller; it is never derived from the
      transaction timestamps and the wall clock is never consulted

    Uniqueness is enforced by the schema-v1 constraints
    (``UNIQUE (chat_id, message_id)`` and ``UNIQUE (update_id)`` on
    ``transactions``, ``PRIMARY KEY (update_id)`` on
    ``processed_updates``); a conflict surfaces as
    :class:`sqlite3.IntegrityError` after a complete rollback. The
    high-level idempotent interpretation belongs to a later slice.

    Returns a new immutable :class:`Transaction` with exactly the same
    business fields as the input and ``transaction_id`` set to the
    string form of the SQLite-assigned ``transactions.id``; the input
    object keeps ``transaction_id is None``.
    """
    _require_connection(connection)
    if not isinstance(transaction, Transaction):
        raise TypeError(
            f"transaction must be a Transaction, got {type(transaction).__name__!r}"
        )
    if not isinstance(provenance, TelegramMessageRef):
        raise TypeError(
            f"provenance must be a TelegramMessageRef, got {type(provenance).__name__!r}"
        )
    if transaction.transaction_id is not None:
        raise RepositoryDataError(
            "transaction is already persisted: transaction_id must be None"
        )
    aware_processed_at = require_aware_datetime(processed_at, "processed_at")
    if connection.in_transaction:
        raise RepositoryTransactionError(
            "connection already has an active transaction; persist_transaction "
            "must own its write transaction"
        )

    connection.execute("BEGIN IMMEDIATE")
    try:
        cursor = connection.execute(
            _INSERT_TRANSACTION_SQL, _transaction_row_values(transaction, provenance)
        )
        row_id = cursor.lastrowid
        if row_id is None or row_id <= 0:
            raise RepositoryTransactionError(
                f"SQLite did not assign a positive transactions row id: {row_id!r}"
            )
        connection.execute(
            _INSERT_PROCESSED_UPDATE_SQL,
            (
                provenance.update_id,
                provenance.chat_id,
                provenance.message_thread_id,
                provenance.message_id,
                aware_processed_at.isoformat(),
            ),
        )
        persisted = Transaction(
            direction=transaction.direction,
            amount_usdt=transaction.amount_usdt,
            category=transaction.category,
            source=transaction.source,
            comment=transaction.comment,
            transaction_date=transaction.transaction_date,
            created_at=transaction.created_at,
            updated_at=transaction.updated_at,
            transaction_id=str(row_id),
            status=transaction.status,
            deleted_at=transaction.deleted_at,
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    return persisted


def get_transaction(
    connection: sqlite3.Connection,
    transaction_id: str,
) -> Transaction | None:
    """Return the persisted :class:`Transaction` with the given ID.

    ``transaction_id`` must be a positive decimal string of the form
    assigned by :func:`persist_transaction` (for example ``"1"`` or
    ``"25"``); malformed IDs and non-string values are rejected
    cleanly. Returns ``None`` when no such transaction exists.

    Read-only: no rows are written and no transaction is started.
    """
    _require_connection(connection)
    numeric_id = _parse_transaction_id(transaction_id)
    row = connection.execute(_SELECT_TRANSACTION_BY_ID, (numeric_id,)).fetchone()
    if row is None:
        return None
    return _row_to_transaction(row)


def find_transaction_by_message(
    connection: sqlite3.Connection,
    identity: TelegramMessageIdentity,
) -> Transaction | None:
    """Return the transaction persisted for a logical Telegram message.

    The lookup uses exactly the logical message identity
    ``(chat_id, message_id)``; ``message_thread_id`` and ``update_id``
    are deliberately not part of the query. Returns ``None`` when no
    such transaction exists. Duplicate interpretation belongs to a
    later slice: here a hit is at most one row by schema constraint.

    Read-only: no rows are written and no transaction is started.
    """
    _require_connection(connection)
    if not isinstance(identity, TelegramMessageIdentity):
        raise TypeError(
            "identity must be a TelegramMessageIdentity, "
            f"got {type(identity).__name__!r}"
        )
    row = connection.execute(
        _SELECT_TRANSACTION_BY_MESSAGE, (identity.chat_id, identity.message_id)
    ).fetchone()
    if row is None:
        return None
    return _row_to_transaction(row)


def find_transaction_by_update(
    connection: sqlite3.Connection,
    identity: TelegramUpdateIdentity,
) -> Transaction | None:
    """Return the transaction persisted for a Telegram update delivery.

    The lookup uses exactly ``update_id``. Returns ``None`` when no such
    transaction exists.

    Read-only: no rows are written and no transaction is started.
    """
    _require_connection(connection)
    if not isinstance(identity, TelegramUpdateIdentity):
        raise TypeError(
            "identity must be a TelegramUpdateIdentity, "
            f"got {type(identity).__name__!r}"
        )
    row = connection.execute(_SELECT_TRANSACTION_BY_UPDATE, (identity.update_id,)).fetchone()
    if row is None:
        return None
    return _row_to_transaction(row)


def is_update_processed(
    connection: sqlite3.Connection,
    identity: TelegramUpdateIdentity,
) -> bool:
    """Return whether a Telegram update delivery was already processed.

    Checks the ``processed_updates`` table by exactly ``update_id``;
    the existence of a ``transactions`` row is never used as a
    substitute.

    Read-only: no rows are written and no transaction is started.
    """
    _require_connection(connection)
    if not isinstance(identity, TelegramUpdateIdentity):
        raise TypeError(
            "identity must be a TelegramUpdateIdentity, "
            f"got {type(identity).__name__!r}"
        )
    row = connection.execute(_SELECT_PROCESSED_UPDATE, (identity.update_id,)).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# bounded stage-C3 primitives
#
# These two functions are deliberately NOT added to ``__all__``: the
# accepted C2 API-surface contract pins that list exactly. They are
# public repository primitives nonetheless and are re-exported by the
# package ``__init__`` for the stage-C3 ingest service.
# ---------------------------------------------------------------------------


def record_processed_update(
    connection: sqlite3.Connection,
    provenance: TelegramMessageRef,
    *,
    processed_at: datetime,
) -> None:
    """Record one processed Telegram update without creating a transaction.

    Exactly one ``processed_updates`` row is written inside one explicit
    ``BEGIN IMMEDIATE`` transaction; no ``transactions`` row is touched.
    Every repository operation after ``BEGIN`` -- including the final
    ``commit`` itself -- is inside the rollback-protected region, so any
    failure after ``BEGIN`` triggers a full rollback before the
    exception propagates and never knowingly leaves the repository's
    own transaction pending.

    Contract (identical to :func:`persist_transaction`):

    - ``connection`` must be a caller-owned
      :class:`sqlite3.Connection` prepared by
      :func:`hermes_finance.storage.open_database`
    - ``connection`` must not already carry an active transaction; the
      repository must own its write transaction so unrelated caller
      work is never silently committed or rolled back
    - ``provenance`` must be a
      :class:`~hermes_finance.provenance.TelegramMessageRef`
    - ``processed_at`` must be a timezone-aware ``datetime`` supplied
      explicitly by the caller; the wall clock is never consulted

    Uniqueness is enforced by ``processed_updates.update_id`` being the
    PRIMARY KEY; a conflict surfaces as :class:`sqlite3.IntegrityError`
    after a complete rollback. There are no retries, no nested
    transactions, and no savepoints: interpreting a lost race belongs
    to the stage-C3 ingest service.
    """
    _require_connection(connection)
    if not isinstance(provenance, TelegramMessageRef):
        raise TypeError(
            f"provenance must be a TelegramMessageRef, got {type(provenance).__name__!r}"
        )
    aware_processed_at = require_aware_datetime(processed_at, "processed_at")
    if connection.in_transaction:
        raise RepositoryTransactionError(
            "connection already has an active transaction; record_processed_update "
            "must own its write transaction"
        )

    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            _INSERT_PROCESSED_UPDATE_SQL,
            (
                provenance.update_id,
                provenance.chat_id,
                provenance.message_thread_id,
                provenance.message_id,
                aware_processed_at.isoformat(),
            ),
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def _row_to_processed_update_ref(row: Sequence[object], update_id: int) -> TelegramMessageRef:
    """Reconstruct a :class:`TelegramMessageRef` from a processed_updates row.

    Any deterministic mapping or validation failure (``TypeError`` for
    wrong stored types such as TEXT or BLOB values in the provenance
    columns, ``ValueError`` for domain-invalid identifiers, and
    ``TypeError``/``ValueError`` for an unexpected row shape) is
    converted into a :class:`RepositoryDataError` with the original
    exception preserved as ``__cause__``. Arbitrary exceptions never
    escape, and non-integer SQLite values are never silently coerced.
    """
    try:
        (
            raw_update_id,
            raw_chat_id,
            raw_thread_id,
            raw_message_id,
        ) = row
        if not isinstance(raw_update_id, int) or isinstance(raw_update_id, bool):
            raise TypeError(
                f"processed_updates.update_id must be an integer, got {raw_update_id!r}"
            )
        if not isinstance(raw_chat_id, int) or isinstance(raw_chat_id, bool):
            raise TypeError(
                f"processed_updates.chat_id must be an integer, got {raw_chat_id!r}"
            )
        if not isinstance(raw_thread_id, int) or isinstance(raw_thread_id, bool):
            raise TypeError(
                f"processed_updates.message_thread_id must be an integer, "
                f"got {raw_thread_id!r}"
            )
        if not isinstance(raw_message_id, int) or isinstance(raw_message_id, bool):
            raise TypeError(
                f"processed_updates.message_id must be an integer, got {raw_message_id!r}"
            )
        return TelegramMessageRef(
            chat_id=raw_chat_id,
            message_thread_id=raw_thread_id,
            message_id=raw_message_id,
            update_id=raw_update_id,
        )
    except (TypeError, ValueError) as error:
        raise RepositoryDataError(
            f"persisted processed_updates row for update_id {update_id!r} cannot be "
            f"mapped to a TelegramMessageRef: {error}"
        ) from error


def get_processed_update_ref(
    connection: sqlite3.Connection,
    identity: TelegramUpdateIdentity,
) -> TelegramMessageRef | None:
    """Return the stored provenance of an already processed update.

    The lookup reads ``processed_updates`` by exactly ``update_id`` and
    reconstructs a
    :class:`~hermes_finance.provenance.TelegramMessageRef` from the
    stored ``chat_id``, ``message_thread_id``, ``message_id``, and
    ``update_id`` columns. Provenance is never inferred from
    ``transactions`` rows, and ``processed_at`` is deliberately not
    exposed. Returns ``None`` when the update was never processed.

    A stored row that cannot form a valid ``TelegramMessageRef`` raises
    a deterministic :class:`RepositoryDataError` with the original
    cause chained.

    Read-only: no rows are written and no transaction is started.
    """
    _require_connection(connection)
    if not isinstance(identity, TelegramUpdateIdentity):
        raise TypeError(
            "identity must be a TelegramUpdateIdentity, "
            f"got {type(identity).__name__!r}"
        )
    row = connection.execute(
        _SELECT_PROCESSED_UPDATE_REF, (identity.update_id,)
    ).fetchone()
    if row is None:
        return None
    return _row_to_processed_update_ref(row, identity.update_id)


# ---------------------------------------------------------------------------
# bounded stage-D1 read primitives
#
# These three functions are deliberately NOT added to ``__all__``: the
# accepted C2 API-surface contract pins that list exactly. They are
# public repository primitives nonetheless and are consumed by the
# stage-D1 operations layer (:mod:`hermes_finance.operations`), which
# owns the user-facing validation contracts (limit bounds, year/month
# policy). All three are read-only, ACTIVE-only, and deterministically
# ordered by ``transaction_date DESC, id DESC``.
# ---------------------------------------------------------------------------


def _require_real_int(value: object, field: str) -> int:
    """Require a real ``int`` (``bool`` is explicitly rejected)."""
    if isinstance(value, bool):
        raise TypeError(f"{field} must not be a bool")
    if not isinstance(value, int):
        raise TypeError(f"{field} must be an int, got {type(value).__name__!r}")
    return value


def _rows_to_transactions(rows: Sequence[Sequence[object]]) -> tuple[Transaction, ...]:
    """Map every selected row through the accepted C2 row mapper."""
    return tuple(_row_to_transaction(row) for row in rows)


def list_active_transactions_recent(
    connection: sqlite3.Connection,
    *,
    limit: int,
) -> tuple[Transaction, ...]:
    """Return up to ``limit`` newest ACTIVE transactions.

    ``limit`` must be a real positive ``int`` (``bool`` is rejected);
    the user-facing 1..100 policy is owned by the operations layer,
    which is the intended caller of this primitive.

    Rows are selected with ``status = 'active'`` and ordered by
    ``transaction_date DESC, id DESC``; ``created_at`` is never an
    ordering authority. Returns ``()`` for an empty ledger.

    Read-only: no rows are written, no transaction is started,
    committed, or rolled back, and a caller-owned active transaction
    is left untouched.
    """
    _require_connection(connection)
    validated_limit = _require_real_int(limit, "limit")
    if validated_limit < 1:
        raise ValueError(f"limit must be at least 1, got {validated_limit!r}")
    rows = connection.execute(
        _SELECT_ACTIVE_RECENT, (TransactionStatus.ACTIVE.value, validated_limit)
    ).fetchall()
    return _rows_to_transactions(rows)


def list_active_transactions_by_date(
    connection: sqlite3.Connection,
    transaction_date: date,
) -> tuple[Transaction, ...]:
    """Return the ACTIVE transactions of one exact business date.

    ``transaction_date`` must be a plain ``datetime.date``; ``datetime``
    values and arbitrary non-date values are rejected cleanly via the
    accepted domain validation. The match is exact on the stored ISO
    ``transaction_date`` text; no surrounding dates are included.

    Rows are ordered by ``id DESC`` within the single date (the
    ``transaction_date DESC`` term is constant here). Returns ``()``
    when no ACTIVE transaction exists on that date.

    Read-only: no rows are written, no transaction is started,
    committed, or rolled back, and a caller-owned active transaction
    is left untouched.
    """
    _require_connection(connection)
    validated_date = require_calendar_date(transaction_date, "transaction_date")
    rows = connection.execute(
        _SELECT_ACTIVE_BY_DATE,
        (TransactionStatus.ACTIVE.value, validated_date.isoformat()),
    ).fetchall()
    return _rows_to_transactions(rows)


def list_active_transactions_in_month(
    connection: sqlite3.Connection,
    *,
    year: int,
    month: int,
) -> tuple[Transaction, ...]:
    """Return the ACTIVE transactions of one calendar month.

    ``year`` must be a real ``int`` in 1..9999 (the years representable
    by :class:`datetime.date`; ``bool`` is rejected) and ``month`` a
    real ``int`` in 1..12. Strings and floats are never coerced.

    The month boundaries are computed exactly with the stdlib
    :func:`calendar.monthrange`: the query is the inclusive range
    ``transaction_date >= <first day> AND transaction_date <= <last
    day>`` in ISO text. This bounded form deliberately avoids
    constructing the first day of the next month, so
    ``year=9999, month=12`` is handled without ever building a year
    10000 date, and the December -> January rollover falls out of the
    exact calendar arithmetic. No approximate day counts are used.

    Rows are ordered by ``transaction_date DESC, id DESC``. Returns
    ``()`` when no ACTIVE transaction exists in that month.

    Read-only: no rows are written, no transaction is started,
    committed, or rolled back, and a caller-owned active transaction
    is left untouched.
    """
    _require_connection(connection)
    validated_year = _require_real_int(year, "year")
    validated_month = _require_real_int(month, "month")
    if not _MIN_MONTH <= validated_month <= _MAX_MONTH:
        raise ValueError(
            f"month must be between 1 and 12, got {validated_month!r}"
        )
    if not _MIN_YEAR <= validated_year <= _MAX_YEAR:
        raise ValueError(
            f"year must be between 1 and 9999, got {validated_year!r}"
        )
    first_day = date(validated_year, validated_month, 1)
    days_in_month = monthrange(validated_year, validated_month)[1]
    last_day = date(validated_year, validated_month, days_in_month)
    rows = connection.execute(
        _SELECT_ACTIVE_IN_MONTH,
        (
            TransactionStatus.ACTIVE.value,
            first_day.isoformat(),
            last_day.isoformat(),
        ),
    ).fetchall()
    return _rows_to_transactions(rows)


# ---------------------------------------------------------------------------
# bounded stage-D2 mutation primitives
#
# These two functions are deliberately NOT added to ``__all__``: the
# accepted C2 API-surface contract pins that list exactly. They are
# public repository primitives nonetheless and are consumed by the
# stage-D2 mutation service (:mod:`hermes_finance.mutations`), which
# owns the user-facing error interpretation (missing target, deleted
# target, idempotent deletion).
#
# Both follow the accepted C2/C3 write-ownership rule: a connection
# that already carries an active transaction is refused before the
# repository begins its own explicit ``BEGIN IMMEDIATE`` transaction,
# and any failure after ``BEGIN`` -- including the final ``commit``
# itself -- triggers a full rollback before the exception propagates.
# The target row is always selected and reconstructed through the
# accepted C2 row mapper *inside* the repository-owned write
# transaction, so no other writer can change it between the state
# inspection and the mutation. There are no retries, no savepoints,
# and no nested transactions.
# ---------------------------------------------------------------------------

#: Replaces exactly the editable business fields of one transaction.
#: The identity, provenance, ``created_at``, ``status``, and
#: ``deleted_at`` columns are deliberately absent from the SET list.
_UPDATE_TRANSACTION_FIELDS_SQL: Final[str] = (
    "UPDATE transactions SET"
    " direction = ?, amount_usdt = ?, category = ?, source = ?, comment = ?,"
    " transaction_date = ?, updated_at = ?"
    " WHERE id = ?"
)

#: Marks exactly the lifecycle columns of one transaction DELETED.
#: Business and provenance columns are deliberately absent.
_MARK_TRANSACTION_DELETED_SQL: Final[str] = (
    "UPDATE transactions SET status = ?, updated_at = ?, deleted_at = ?"
    " WHERE id = ?"
)


def replace_active_transaction_fields(
    connection: sqlite3.Connection,
    transaction_id: str,
    *,
    direction: Direction,
    amount_usdt: Decimal | int | str,
    category: str,
    source: str,
    comment: str | None,
    transaction_date: date,
    updated_at: datetime,
) -> Transaction | None:
    """Atomically replace the business fields of one ACTIVE transaction.

    The target row is selected and reconstructed inside one explicit
    ``BEGIN IMMEDIATE`` transaction owned by the repository; the
    authoritative :class:`~hermes_finance.domain.Transaction`
    constructor validates the replacement values *after* ``BEGIN``, so
    a domain validation failure triggers the full rollback path.

    Outcomes:

    - target row missing: a deterministic no-write outcome; the
      read-only transaction is committed and ``None`` is returned for
      the caller to interpret
    - target row DELETED: a deterministic no-write outcome; the
      existing DELETED :class:`~hermes_finance.domain.Transaction` is
      returned unchanged for the caller to interpret (deleted history
      is never edited here)
    - target row ACTIVE: an edited immutable
      :class:`~hermes_finance.domain.Transaction` is constructed with
      the replacement business fields, the caller-supplied
      ``updated_at``, the preserved ``created_at`` and
      ``transaction_id``, ``status`` ACTIVE and ``deleted_at`` ``None``

    The ``UPDATE`` statement touches exactly the editable business
    columns (``direction``, ``amount_usdt``, ``category``, ``source``,
    ``comment``, ``transaction_date``, ``updated_at``); the identity,
    provenance (``chat_id``, ``message_thread_id``, ``message_id``,
    ``update_id``), ``created_at``, ``status``, and ``deleted_at``
    columns are never modified, and the ``processed_updates`` table is
    never touched.

    Contract (identical to :func:`persist_transaction`):

    - ``connection`` must be a caller-owned
      :class:`sqlite3.Connection` prepared by
      :func:`hermes_finance.storage.open_database`
    - ``transaction_id`` must be a positive decimal string of the form
      assigned by :func:`persist_transaction`; the accepted repository
      ID validation is authoritative
    - ``updated_at`` must be a timezone-aware ``datetime`` supplied
      explicitly by the caller; it is used exactly as given and the
      wall clock is never consulted
    - ``connection`` must not already carry an active transaction;
      caller-owned work is never committed or rolled back

    Business-field values are validated only through the authoritative
    domain validation (:func:`~hermes_finance.domain.normalize_usdt_amount`
    and the :class:`~hermes_finance.domain.Transaction` constructor
    itself); a ``TypeError``/``ValueError`` from them propagates
    unchanged after the rollback. A corrupt target row raises
    :class:`RepositoryDataError` with the original cause chained, also
    after a full rollback.
    """
    _require_connection(connection)
    numeric_id = _parse_transaction_id(transaction_id)
    aware_updated_at = require_aware_datetime(updated_at, "updated_at")
    if connection.in_transaction:
        raise RepositoryTransactionError(
            "connection already has an active transaction; "
            "replace_active_transaction_fields must own its write transaction"
        )

    connection.execute("BEGIN IMMEDIATE")
    try:
        row = connection.execute(_SELECT_TRANSACTION_BY_ID, (numeric_id,)).fetchone()
        if row is None:
            connection.commit()
            return None
        existing = _row_to_transaction(row)
        if existing.status is TransactionStatus.DELETED:
            connection.commit()
            return existing
        edited = Transaction(
            direction=direction,
            amount_usdt=normalize_usdt_amount(amount_usdt),
            category=category,
            source=source,
            comment=comment,
            transaction_date=transaction_date,
            created_at=existing.created_at,
            updated_at=aware_updated_at,
            transaction_id=existing.transaction_id,
            status=TransactionStatus.ACTIVE,
            deleted_at=None,
        )
        connection.execute(
            _UPDATE_TRANSACTION_FIELDS_SQL,
            (
                edited.direction.value,
                str(edited.amount_usdt),
                edited.category,
                edited.source,
                edited.comment,
                edited.transaction_date.isoformat(),
                edited.updated_at.isoformat(),
                numeric_id,
            ),
        )
        connection.commit()
        return edited
    except BaseException:
        connection.rollback()
        raise


def mark_transaction_deleted(
    connection: sqlite3.Connection,
    transaction_id: str,
    *,
    deleted_at: datetime,
) -> Transaction | None:
    """Atomically mark one transaction DELETED (soft delete).

    The target row is selected and reconstructed inside one explicit
    ``BEGIN IMMEDIATE`` transaction owned by the repository.

    Outcomes:

    - target row missing: a deterministic no-write outcome; the
      read-only transaction is committed and ``None`` is returned for
      the caller to interpret
    - target row already DELETED: a deterministic idempotent no-write
      outcome; the existing DELETED
      :class:`~hermes_finance.domain.Transaction` is returned exactly
      as stored (``updated_at`` and ``deleted_at`` are never
      rewritten), which makes deletion retry-safe
    - target row ACTIVE: a deleted immutable
      :class:`~hermes_finance.domain.Transaction` is constructed from
      the preserved business fields with ``status`` DELETED and both
      ``updated_at`` and ``deleted_at`` set to the caller-supplied
      ``deleted_at``

    The ``UPDATE`` statement touches exactly ``status``, ``updated_at``,
    and ``deleted_at``; every business column, the identity, and the
    provenance (``chat_id``, ``message_thread_id``, ``message_id``,
    ``update_id``) columns are never modified, and the
    ``processed_updates`` table is never touched: soft-deleting a
    financial transaction never un-processes its Telegram delivery.

    Contract (identical to :func:`persist_transaction`):

    - ``connection`` must be a caller-owned
      :class:`sqlite3.Connection` prepared by
      :func:`hermes_finance.storage.open_database`
    - ``transaction_id`` must be a positive decimal string of the form
      assigned by :func:`persist_transaction`; the accepted repository
      ID validation is authoritative
    - ``deleted_at`` must be a timezone-aware ``datetime`` supplied
      explicitly by the caller; it is validated before any write
      begins and used exactly as given, and the wall clock is never
      consulted
    - ``connection`` must not already carry an active transaction;
      caller-owned work is never committed or rolled back

    A corrupt target row raises :class:`RepositoryDataError` with the
    original cause chained, after a full rollback.
    """
    _require_connection(connection)
    numeric_id = _parse_transaction_id(transaction_id)
    aware_deleted_at = require_aware_datetime(deleted_at, "deleted_at")
    if connection.in_transaction:
        raise RepositoryTransactionError(
            "connection already has an active transaction; "
            "mark_transaction_deleted must own its write transaction"
        )

    connection.execute("BEGIN IMMEDIATE")
    try:
        row = connection.execute(_SELECT_TRANSACTION_BY_ID, (numeric_id,)).fetchone()
        if row is None:
            connection.commit()
            return None
        existing = _row_to_transaction(row)
        if existing.status is TransactionStatus.DELETED:
            connection.commit()
            return existing
        deleted = Transaction(
            direction=existing.direction,
            amount_usdt=existing.amount_usdt,
            category=existing.category,
            source=existing.source,
            comment=existing.comment,
            transaction_date=existing.transaction_date,
            created_at=existing.created_at,
            updated_at=aware_deleted_at,
            transaction_id=existing.transaction_id,
            status=TransactionStatus.DELETED,
            deleted_at=aware_deleted_at,
        )
        connection.execute(
            _MARK_TRANSACTION_DELETED_SQL,
            (
                TransactionStatus.DELETED.value,
                aware_deleted_at.isoformat(),
                aware_deleted_at.isoformat(),
                numeric_id,
            ),
        )
        connection.commit()
        return deleted
    except BaseException:
        connection.rollback()
        raise
