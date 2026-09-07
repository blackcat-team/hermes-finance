"""Amount-only transaction correction orchestration for Hermes Finance (stage H1).

This module is the thin stage-H1 orchestration facade over the accepted
D2 mutation core. It exposes exactly one function:

- :func:`edit_transaction_amount`: replace only the ``amount_usdt`` of
  one persisted ACTIVE transaction while preserving every other
  business field

The composition is deliberately minimal:

- stage C2: :func:`hermes_finance.repository.get_transaction` reads the
  current transaction
- stage D2: :func:`hermes_finance.mutations.edit_transaction` performs
  the full-replacement write

The accepted D2 mutation service remains the single write authority:
transaction ID validation, ACTIVE/deleted checks, full replacement,
atomic writes, rollback, provenance preservation, and
``processed_updates`` preservation all stay there. There is no second
mutation implementation, no competing SQL, and no patch API here.

Contract highlights:

- ``amount_usdt`` is the positive magnitude; the stored income/expense
  direction is preserved exactly, never silently flipped (an existing
  ``-10`` expense edited to ``12`` remains an expense of ``12``)
- every other editable business field (``direction``, ``category``,
  ``source``, ``comment``, ``transaction_date``) is passed through from
  the currently persisted transaction, so the D2 full-replacement seam
  receives the complete desired field set
- the loaded :class:`~hermes_finance.domain.Transaction` is never
  mutated; the accepted D2 service re-reads the target row inside its
  own write transaction and is the sole authority for the persisted
  result
- provenance is never rebuilt and ``processed_updates`` is never
  touched: the accepted D2 semantics are preserved unchanged
- ``updated_at`` is always supplied explicitly by the caller as a
  timezone-aware datetime; the wall clock is never consulted
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from decimal import Decimal

from hermes_finance.domain import Transaction
from hermes_finance.mutations import TransactionNotFoundError, edit_transaction
from hermes_finance.repository import get_transaction

__all__ = ["edit_transaction_amount"]


def edit_transaction_amount(
    connection: sqlite3.Connection,
    transaction_id: str,
    *,
    amount_usdt: Decimal | int | str,
    updated_at: datetime,
) -> Transaction:
    """Replace only the amount of one persisted ACTIVE transaction.

    This is the stage-H1 amount-only correction seam: the caller
    supplies the target ``transaction_id``, the new positive
    ``amount_usdt`` magnitude, and the timezone-aware ``updated_at``
    timestamp; the current transaction is read through the accepted
    repository primitive and the complete desired business-field set
    (the new amount plus every other current field) is delegated to the
    accepted D2 full-replacement mutation, which remains the single
    write authority.

    Replaced fields:

    - ``amount_usdt`` (the caller-supplied magnitude)
    - ``updated_at`` (used exactly as supplied by the caller)

    Preserved exactly (read from the current transaction and passed
    through unchanged):

    - ``direction`` (an expense stays an expense, an income stays an
      income; the amount never carries the direction)
    - ``category``
    - ``source``
    - ``comment``
    - ``transaction_date``
    - ``transaction_id``, ``created_at``, the ACTIVE status, the
      persisted Telegram provenance, and the ``processed_updates``
      table (all preserved by the accepted D2 mutation itself)

    Contract:

    - ``connection`` must be a real caller-owned
      :class:`sqlite3.Connection` prepared by
      :func:`hermes_finance.storage.open_database`; the accepted
      lower-layer connection and ownership contracts remain
      authoritative
    - ``transaction_id`` follows the accepted repository ID contract;
      the authoritative repository validation is reused and there is no
      competing ID parser
    - ``amount_usdt`` is validated only through the authoritative
      domain amount validation inside the accepted D2 mutation; a
      negative, zero, or otherwise invalid amount is rejected there
      with no partial mutation surviving
    - the loaded current transaction is never mutated: only its field
      values are read and handed to the accepted mutation

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
        input contract and the authoritative domain validation of the
        accepted lower layers.
    """

    current = get_transaction(connection, transaction_id)
    if current is None:
        raise TransactionNotFoundError(
            f"no persisted transaction with transaction_id {transaction_id!r}"
        )
    return edit_transaction(
        connection,
        transaction_id,
        direction=current.direction,
        amount_usdt=amount_usdt,
        category=current.category,
        source=current.source,
        comment=current.comment,
        transaction_date=current.transaction_date,
        updated_at=updated_at,
    )
