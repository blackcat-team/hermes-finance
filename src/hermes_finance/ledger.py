"""Deterministic transaction-creation core for Hermes Finance (stage B2).

This module bridges the parsing layer and the domain layer: it converts an
already validated :class:`hermes_finance.parser.ParsedTransactionInput`
into a complete, validated, immutable
:class:`hermes_finance.domain.Transaction`.

The module is deliberately free of:

- wall-clock access: ``transaction_date`` and ``created_at`` are always
  supplied explicitly by the caller; the later Hermes runtime integration
  will decide business date/time using ``FinanceConfig.business_timezone``
- persistence: no IDs are generated (``transaction_id`` stays ``None``
  until the persistence layer assigns it), and nothing is stored,
  listed, updated, or deleted
- Telegram provenance: ``TelegramMessageRef`` is joined with the
  transaction only in the future persistence/ledger-storage layer

All financial invariants remain authoritative in the domain layer: this
module delegates every value through the accepted :class:`Transaction`
constructor instead of re-validating them.
"""

from __future__ import annotations

from datetime import date, datetime

from hermes_finance.domain import Transaction, TransactionStatus
from hermes_finance.parser import ParsedTransactionInput

__all__ = [
    "create_transaction",
]


def create_transaction(
    parsed: ParsedTransactionInput,
    *,
    transaction_date: date,
    created_at: datetime,
) -> Transaction:
    """Create an initial ACTIVE :class:`Transaction` from parsed input.

    The parsed values are copied field-by-field through the authoritative
    :class:`Transaction` domain constructor, so every domain invariant
    (money, text, date and datetime rules) is enforced exactly once, in the
    domain layer. The ``ParsedTransactionInput`` is never mutated.

    Parameters
    ----------
    parsed:
        A validated :class:`ParsedTransactionInput` (for example produced
        by :func:`hermes_finance.parser.parse_transaction_input`).
    transaction_date:
        The business date of the transaction, supplied explicitly by the
        caller; it is used exactly as given and never derived from
        ``created_at``.
    created_at:
        The timezone-aware creation timestamp, supplied explicitly by the
        caller; it is never silently converted to another timezone and
        also becomes the initial ``updated_at``.

    Returns
    -------
    Transaction
        A new immutable transaction with ``status`` ``ACTIVE``,
        ``transaction_id`` ``None`` (assigned only later by persistence),
        and ``deleted_at`` ``None``.

    Raises
    ------
    TypeError
        If ``parsed`` is not a :class:`ParsedTransactionInput`, or if the
        supplied date/time values have an invalid type (as determined by
        the domain constructor).
    ValueError
        If the supplied date/time values violate domain rules (for
        example a timezone-naive ``created_at``).
    """
    if not isinstance(parsed, ParsedTransactionInput):
        raise TypeError(
            f"parsed must be a ParsedTransactionInput, got {type(parsed).__name__!r}"
        )

    return Transaction(
        direction=parsed.direction,
        amount_usdt=parsed.amount_usdt,
        category=parsed.category,
        source=parsed.source,
        comment=parsed.comment,
        transaction_date=transaction_date,
        created_at=created_at,
        updated_at=created_at,
        transaction_id=None,
        status=TransactionStatus.ACTIVE,
        deleted_at=None,
    )
