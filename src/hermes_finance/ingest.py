"""Semantic idempotent ingest service for Hermes Finance (stage C3).

This module interprets Telegram retries and redeliveries on top of the
accepted stage-C2 repository primitives without ever creating a second
financial transaction for the same logical Telegram message. Logical
financial identity remains ``(chat_id, message_id)``;
``message_thread_id`` and ``update_id`` are delivery context only.

Dispositions
------------

- :attr:`IngestDisposition.CREATED`: a brand-new logical message and
  update; :func:`hermes_finance.repository.persist_transaction`
  atomically persisted both rows
- :attr:`IngestDisposition.DUPLICATE_MESSAGE`: the same logical message
  arrived through a new ``update_id``; no second ``transactions`` row
  is created, the new delivery is recorded in ``processed_updates``,
  and the existing persisted transaction is returned
- :attr:`IngestDisposition.DUPLICATE_UPDATE`: an already processed
  ``update_id`` was replayed; nothing is written and the transaction of
  the stored logical message is returned (the replayed update may have
  no ``transactions`` row carrying its own ``update_id``)

Failure semantics
-----------------

- schema uniqueness remains the race-safety authority: a
  :class:`sqlite3.IntegrityError` from a persist attempt is
  interpreted as an idempotency conflict only when post-failure reads
  prove it, and an unproven conflict is re-raised unchanged
- :class:`IngestConsistencyError` signals logically impossible
  persisted state: a stored processed update whose provenance conflicts
  with the incoming delivery, or an orphan processed update whose
  stored logical message has no matching ``transactions`` row
- classification reads are side-effect free; the only writes happen
  through the explicit repository write primitives
- the wall clock is never consulted: ``processed_at`` always comes from
  the caller
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from hermes_finance.domain import Transaction, require_aware_datetime
from hermes_finance.provenance import (
    TelegramMessageIdentity,
    TelegramMessageRef,
    TelegramUpdateIdentity,
)
from hermes_finance.repository import (
    RepositoryDataError,
    find_transaction_by_message,
    get_processed_update_ref,
    persist_transaction,
    record_processed_update,
)

__all__ = [
    "IngestConsistencyError",
    "IngestDisposition",
    "IngestResult",
    "ingest_transaction",
]


class IngestDisposition(Enum):
    """Deterministic outcome of one idempotent ingest attempt."""

    CREATED = "created"
    DUPLICATE_MESSAGE = "duplicate_message"
    DUPLICATE_UPDATE = "duplicate_update"


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Immutable ingest outcome: the disposition and the transaction.

    ``transaction`` is always a *persisted*
    :class:`~hermes_finance.domain.Transaction`: for ``CREATED`` it is
    the newly persisted copy with its assigned ``transaction_id``; for
    both duplicate dispositions it is the existing persisted
    transaction of the logical Telegram message.
    """

    disposition: IngestDisposition
    transaction: Transaction


class IngestConsistencyError(Exception):
    """Raised when persisted provenance or state is logically impossible.

    Covers two deterministic cases:

    - a stored processed update whose provenance conflicts with the
      incoming delivery: the same ``update_id`` already stored with a
      different ``chat_id``, ``message_thread_id``, or ``message_id``
    - an orphan processed update whose stored logical message has no
      matching ``transactions`` row

    The persisted state is never silently repaired or worked around:
    the inconsistency is surfaced to the caller.
    """


def _provenance_matches(stored: TelegramMessageRef, incoming: TelegramMessageRef) -> bool:
    """Exact equality of update/chat/thread/message provenance fields."""
    return (
        stored.update_id == incoming.update_id
        and stored.chat_id == incoming.chat_id
        and stored.message_thread_id == incoming.message_thread_id
        and stored.message_id == incoming.message_id
    )


def _provenance_conflict_message(
    stored: TelegramMessageRef, incoming: TelegramMessageRef
) -> str:
    """Deterministic message for a conflicting processed update."""
    return (
        f"processed update {incoming.update_id} is already stored with different "
        f"provenance: stored chat_id={stored.chat_id}, "
        f"message_thread_id={stored.message_thread_id}, "
        f"message_id={stored.message_id}; incoming chat_id={incoming.chat_id}, "
        f"message_thread_id={incoming.message_thread_id}, "
        f"message_id={incoming.message_id}"
    )


def _orphan_message(stored: TelegramMessageRef) -> str:
    """Deterministic message for an orphan processed update."""
    return (
        f"processed update {stored.update_id} references logical message "
        f"({stored.chat_id}, {stored.message_id}) but no persisted transaction "
        f"exists for it"
    )


def _classify_processed_update(
    connection: sqlite3.Connection,
    incoming: TelegramMessageRef,
    stored: TelegramMessageRef,
) -> IngestResult:
    """Classify an already-processed update as a duplicate delivery.

    The stored provenance must match the incoming delivery exactly;
    otherwise the state is inconsistent. The existing financial
    transaction is located through the *stored* logical message
    identity, because a duplicate-message update has no ``transactions``
    row carrying its own ``update_id``.
    """
    if not _provenance_matches(stored, incoming):
        raise IngestConsistencyError(_provenance_conflict_message(stored, incoming))
    existing = find_transaction_by_message(
        connection,
        TelegramMessageIdentity(chat_id=stored.chat_id, message_id=stored.message_id),
    )
    if existing is None:
        raise IngestConsistencyError(_orphan_message(stored))
    return IngestResult(
        disposition=IngestDisposition.DUPLICATE_UPDATE,
        transaction=existing,
    )


def _interpret_record_conflict(
    connection: sqlite3.Connection,
    provenance: TelegramMessageRef,
    error: sqlite3.IntegrityError,
) -> IngestResult:
    """Classify a lost race while recording a duplicate-message update.

    ``record_processed_update`` failed with an
    :class:`sqlite3.IntegrityError` and has already rolled back. If the
    update is still absent the conflict is unproven and the original
    error is re-raised. Otherwise another writer recorded this exact
    delivery first and the outcome is classified from the stored row.
    """
    stored = get_processed_update_ref(
        connection, TelegramUpdateIdentity(update_id=provenance.update_id)
    )
    if stored is None:
        raise error
    try:
        return _classify_processed_update(connection, provenance, stored)
    except IngestConsistencyError as consistency_error:
        raise consistency_error from error


def _interpret_persist_conflict(
    connection: sqlite3.Connection,
    provenance: TelegramMessageRef,
    processed_at: datetime,
    error: sqlite3.IntegrityError,
) -> IngestResult:
    """Classify a failed persist attempt through post-failure reads.

    The failed ``persist_transaction`` has already fully rolled back, so
    the reads below observe clean persisted state.

    First the incoming ``update_id`` is re-read: if another writer won a
    race processing the same update, the stored provenance is validated
    and the update is classified as a duplicate delivery.

    Otherwise the incoming logical message identity is checked: an
    existing transaction means a logical-message redelivery via a new
    update, so only the incoming update is recorded (a
    ``DUPLICATE_MESSAGE`` outcome). If neither exists, the conflict is
    not established as an idempotency duplicate and the original
    :class:`sqlite3.IntegrityError` is re-raised unchanged.
    """
    stored = get_processed_update_ref(
        connection, TelegramUpdateIdentity(update_id=provenance.update_id)
    )
    if stored is not None:
        try:
            return _classify_processed_update(connection, provenance, stored)
        except IngestConsistencyError as consistency_error:
            raise consistency_error from error

    existing = find_transaction_by_message(
        connection,
        TelegramMessageIdentity(
            chat_id=provenance.chat_id, message_id=provenance.message_id
        ),
    )
    if existing is None:
        raise error

    try:
        record_processed_update(connection, provenance, processed_at=processed_at)
    except sqlite3.IntegrityError as record_error:
        return _interpret_record_conflict(connection, provenance, record_error)
    return IngestResult(
        disposition=IngestDisposition.DUPLICATE_MESSAGE,
        transaction=existing,
    )


def ingest_transaction(
    connection: sqlite3.Connection,
    transaction: Transaction,
    provenance: TelegramMessageRef,
    *,
    processed_at: datetime,
) -> IngestResult:
    """Ingest one Telegram-originated transaction idempotently.

    Contract:

    - ``connection`` must be a real caller-owned
      :class:`sqlite3.Connection` prepared by
      :func:`hermes_finance.storage.open_database`
    - ``transaction`` must be an unpersisted
      :class:`~hermes_finance.domain.Transaction`
      (``transaction_id is None``); the input object is never mutated
      and keeps ``transaction_id is None``
    - ``provenance`` must be a
      :class:`~hermes_finance.provenance.TelegramMessageRef`
    - ``processed_at`` must be a timezone-aware ``datetime`` supplied
      explicitly by the caller; the wall clock is never consulted
    - no ``FinanceConfig`` is accepted: configuration orchestration
      belongs to later stages

    Outcomes:

    - brand-new logical message + update -> ``CREATED`` with the newly
      persisted transaction
    - same logical message through a new ``update_id`` ->
      ``DUPLICATE_MESSAGE``: no second financial transaction is
      created, the new delivery is recorded in ``processed_updates``,
      and the existing persisted transaction is returned
    - already processed ``update_id`` replay -> ``DUPLICATE_UPDATE``:
      zero writes, and the transaction of the stored logical message is
      returned

    Raises
    ------
    IngestConsistencyError
        If a stored processed update conflicts with the incoming
        provenance, or an orphan processed update has no matching
        transaction.
    sqlite3.IntegrityError
        If a persist conflict cannot be proven to be a valid
        idempotency case; the original error is re-raised unchanged.
    RepositoryTransactionError
        If a write is required while the caller already owns an active
        transaction.
    RepositoryDataError
        If the transaction is already persisted or a persisted row is
        corrupt.
    """
    if not isinstance(connection, sqlite3.Connection):
        raise TypeError(
            f"connection must be a sqlite3.Connection, got {type(connection).__name__!r}"
        )
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
    require_aware_datetime(processed_at, "processed_at")

    # Fast path: the delivering update may already be processed. All
    # reads here are side-effect free.
    stored = get_processed_update_ref(
        connection, TelegramUpdateIdentity(update_id=provenance.update_id)
    )
    if stored is not None:
        return _classify_processed_update(connection, provenance, stored)

    try:
        persisted = persist_transaction(
            connection, transaction, provenance, processed_at=processed_at
        )
    except sqlite3.IntegrityError as error:
        return _interpret_persist_conflict(connection, provenance, processed_at, error)
    return IngestResult(disposition=IngestDisposition.CREATED, transaction=persisted)
