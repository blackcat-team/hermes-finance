"""Deterministic transaction mutation operations for Hermes Finance (stage D2).

This module is the deliberate mutation layer between Hermes (the future
Telegram-facing component) and the accepted repository primitives. It
exposes exactly two function-based mutations and their bounded error
types:

- :func:`edit_transaction`: full replacement of the editable business
  fields of one ACTIVE transaction
- :func:`soft_delete_transaction`: idempotent soft deletion of one
  transaction

Contract highlights:

- every operation requires a real caller-owned
  :class:`sqlite3.Connection` prepared by
  :func:`hermes_finance.storage.open_database`; lookalike values are
  rejected cleanly with a :class:`TypeError`
- ``transaction_id`` follows the accepted repository ID contract (a
  positive decimal string); the authoritative repository validation is
  reused and there is no competing ID parser
- edit is a *full replacement* of the editable business fields
  (``direction``, ``amount_usdt``, ``category``, ``source``,
  ``comment``, ``transaction_date``); there is deliberately no
  patch/sentinel API, so "comment unchanged" and "comment
  intentionally set to None" are unambiguous -- a later caller can
  read the current transaction, replace only the user-requested
  values, and pass the complete desired business-field set here
- ``updated_at`` and ``deleted_at`` are always supplied explicitly by
  the caller as timezone-aware datetimes; the wall clock is never
  consulted
- edit preserves the transaction identity, ``created_at``, the ACTIVE
  status, and the persisted Telegram provenance; soft delete preserves
  every business field and the provenance
- ``processed_updates`` is never touched: mutating a financial
  transaction never un-processes its Telegram delivery
- all business-field validation is delegated to the authoritative
  :class:`~hermes_finance.domain.Transaction` constructor; no domain
  rule is re-implemented here
- every successful result is an immutable persisted
  :class:`~hermes_finance.domain.Transaction`
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from decimal import Decimal

from hermes_finance.domain import Direction, Transaction, TransactionStatus
from hermes_finance.repository import (
    mark_transaction_deleted,
    replace_active_transaction_fields,
)

__all__ = [
    "TransactionMutationError",
    "TransactionNotActiveError",
    "TransactionNotFoundError",
    "edit_transaction",
    "soft_delete_transaction",
]


class TransactionMutationError(Exception):
    """Base class for deterministic transaction mutation failures.

    Deliberately distinct from the repository error types
    (:class:`~hermes_finance.repository.RepositoryDataError`,
    :class:`~hermes_finance.repository.RepositoryTransactionError`)
    and from raw :class:`sqlite3.Error`: repository corruption,
    connection-ownership violations, and SQLite failures are never
    collapsed into mutation errors.
    """


class TransactionNotFoundError(TransactionMutationError):
    """The target ``transaction_id`` does not identify a persisted transaction."""


class TransactionNotActiveError(TransactionMutationError):
    """The target transaction exists but is not ACTIVE.

    Editing deleted history is refused with this error. Soft-deleting
    an already DELETED transaction is *not* an error: it is an
    idempotent no-write outcome and never raises this error.
    """


def edit_transaction(
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
) -> Transaction:
    """Replace the editable business fields of one ACTIVE transaction.

    This is a full replacement of the editable business fields, not a
    patch: the caller supplies the complete desired field set (a later
    Hermes layer can read the current transaction and substitute only
    the user-requested values). An edit whose business values equal the
    current ones is still a valid edit: ``updated_at`` becomes the
    caller-supplied value.

    Replaced fields:

    - ``direction``
    - ``amount_usdt``
    - ``category``
    - ``source``
    - ``comment`` (``None`` explicitly clears it)
    - ``transaction_date``
    - ``updated_at`` (used exactly as supplied by the caller)

    Preserved exactly:

    - ``transaction_id``
    - ``created_at``
    - ``status`` (ACTIVE) and ``deleted_at`` (``None``)
    - the persisted Telegram provenance (``chat_id``,
      ``message_thread_id``, ``message_id``, ``update_id``)
    - the ``processed_updates`` table

    Contract:

    - ``connection`` must be a real caller-owned
      :class:`sqlite3.Connection`; lookalike values are rejected with
      a :class:`TypeError`
    - ``transaction_id`` must be a positive decimal string of the form
      assigned by the repository; the accepted repository ID
      validation is authoritative
    - every business value is validated only through the authoritative
      :class:`~hermes_finance.domain.Transaction` constructor; the
      resulting ``TypeError``/``ValueError`` propagates unchanged and
      no partial mutation survives
    - the target row is read, reconstructed, and updated inside one
      repository-owned ``BEGIN IMMEDIATE`` write transaction; any
      failure after ``BEGIN`` -- including the final commit -- rolls
      the repository transaction back before propagating

    Raises
    ------
    TransactionNotFoundError
        If ``transaction_id`` is valid but no such transaction exists.
    TransactionNotActiveError
        If the target transaction exists but is DELETED.
    RepositoryTransactionError
        If the connection already carries an active caller-owned
        transaction.
    RepositoryDataError
        If the persisted target row is corrupt.
    TypeError / ValueError
        For invalid argument values, as determined by the repository
        input contract and the authoritative domain constructor.
    """

    result = replace_active_transaction_fields(
        connection,
        transaction_id,
        direction=direction,
        amount_usdt=amount_usdt,
        category=category,
        source=source,
        comment=comment,
        transaction_date=transaction_date,
        updated_at=updated_at,
    )
    if result is None:
        raise TransactionNotFoundError(
            f"no persisted transaction with transaction_id {transaction_id!r}"
        )
    if result.status is TransactionStatus.DELETED:
        raise TransactionNotActiveError(
            f"transaction {transaction_id!r} is DELETED and cannot be edited"
        )
    return result


def soft_delete_transaction(
    connection: sqlite3.Connection,
    transaction_id: str,
    *,
    deleted_at: datetime,
) -> Transaction:
    """Soft-delete one transaction (idempotent).

    For an ACTIVE target, every business field, the identity,
    ``created_at``, and the persisted Telegram provenance are
    preserved, and the row is atomically marked DELETED with:

    - ``status`` = DELETED
    - ``updated_at`` = the caller-supplied ``deleted_at``
    - ``deleted_at`` = the caller-supplied ``deleted_at``

    For an already DELETED target, this is an idempotent no-write
    operation: the existing DELETED
    :class:`~hermes_finance.domain.Transaction` is returned exactly as
    stored (``deleted_at`` and ``updated_at`` keep their original
    values) and the row is not touched again, which makes deletion
    retry-safe. There is no hard delete and no restore.

    The ``processed_updates`` table is never touched: soft-deleting a
    financial transaction never un-processes its Telegram delivery, so
    a later replay of the original update remains a processed delivery.

    Contract:

    - ``connection`` must be a real caller-owned
      :class:`sqlite3.Connection`; lookalike values are rejected with
      a :class:`TypeError`
    - ``transaction_id`` must be a positive decimal string of the form
      assigned by the repository; the accepted repository ID
      validation is authoritative
    - ``deleted_at`` must be a timezone-aware ``datetime`` supplied
      explicitly by the caller; it is validated before any write
      begins (also on the idempotent path) and the wall clock is never
      consulted
    - the target row is read, reconstructed, and updated inside one
      repository-owned ``BEGIN IMMEDIATE`` write transaction; any
      failure after ``BEGIN`` -- including the final commit -- rolls
      the repository transaction back before propagating

    Raises
    ------
    TransactionNotFoundError
        If ``transaction_id`` is valid but no such transaction exists.
    RepositoryTransactionError
        If the connection already carries an active caller-owned
        transaction.
    RepositoryDataError
        If the persisted target row is corrupt.
    TypeError / ValueError
        For invalid argument values, as determined by the repository
        input contract and domain timestamp validation.
    """

    result = mark_transaction_deleted(connection, transaction_id, deleted_at=deleted_at)
    if result is None:
        raise TransactionNotFoundError(
            f"no persisted transaction with transaction_id {transaction_id!r}"
        )
    return result
