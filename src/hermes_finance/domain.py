"""Pure domain model for Hermes Finance.

This module is the isolated domain foundation of Hermes Finance: USDT money
handling with arbitrary ``Decimal`` precision, transaction direction and
lifecycle status, text normalisation, timezone-aware timestamp validation,
and the immutable :class:`Transaction` entity.

The module deliberately contains no persistence, parsing, scheduling,
reporting, or transport logic; those concerns belong to later slices.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Final

__all__ = [
    "Direction",
    "Transaction",
    "TransactionStatus",
    "normalize_optional_text",
    "normalize_required_text",
    "normalize_usdt_amount",
    "require_aware_datetime",
    "require_calendar_date",
]


class Direction(Enum):
    """Direction of a money movement.

    Direction is modelled separately from the amount: the amount is always
    a positive USDT value and the direction carries the income/expense
    semantics.
    """

    INCOME = "income"
    EXPENSE = "expense"


class TransactionStatus(Enum):
    """Lifecycle status of a transaction."""

    ACTIVE = "active"
    DELETED = "deleted"


# A clean decimal string: optional leading "+", digits, optional fraction.
# Scientific notation, thousands separators, signs other than a leading "+",
# and any other stray characters are rejected.
_DECIMAL_STRING_RE: Final[re.Pattern[str]] = re.compile(r"\+?[0-9]+(?:\.[0-9]+)?")

_ZERO: Final[Decimal] = Decimal(0)


def normalize_usdt_amount(value: object) -> Decimal:
    """Normalise and validate a USDT amount, preserving full precision.

    Accepted inputs:

    - :class:`decimal.Decimal` (finite and positive)
    - ``int`` (positive), converted exactly
    - ``str`` in a clean decimal form such as ``"25"``, ``"0.1"``,
      ``"123.456789"`` (surrounding whitespace is stripped)

    Rejected inputs:

    - ``bool`` (even though ``bool`` is an ``int`` subclass)
    - ``float`` (floats must never be used for money)
    - zero, negative, ``NaN`` and infinite values
    - any other type, or a string that is not a clean decimal

    The returned ``Decimal`` is never quantised: arbitrary non-zero
    precision is preserved exactly. There is deliberately no minor-unit
    conversion and no fixed number of decimal places.
    """
    if isinstance(value, bool):
        raise TypeError("USDT amount must not be a bool")
    if isinstance(value, float):
        raise TypeError("USDT amount must not be a float; use Decimal, int or a decimal string")
    if isinstance(value, int):
        amount = Decimal(value)
    elif isinstance(value, Decimal):
        amount = value
    elif isinstance(value, str):
        text = value.strip()
        if _DECIMAL_STRING_RE.fullmatch(text) is None:
            raise ValueError(f"USDT amount is not a clean decimal string: {value!r}")
        amount = Decimal(text)
    else:
        raise TypeError(f"USDT amount has unsupported type {type(value).__name__!r}")

    if not amount.is_finite():
        raise ValueError("USDT amount must be a finite number")
    if amount <= _ZERO:
        raise ValueError("USDT amount must be positive")
    return amount


def normalize_required_text(value: object, field: str) -> str:
    """Strip surrounding whitespace and require a non-empty string."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must not be empty")
    return text


def normalize_optional_text(value: object, field: str) -> str | None:
    """Strip surrounding whitespace; blank values normalise to ``None``."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string or None")
    text = value.strip()
    return text or None


def require_aware_datetime(value: object, field: str) -> datetime:
    """Require a timezone-aware ``datetime``; naive datetimes are rejected."""
    if not isinstance(value, datetime):
        raise TypeError(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value


def require_calendar_date(value: object, field: str) -> date:
    """Require a plain calendar ``date``; ``datetime`` values are rejected."""
    if isinstance(value, datetime):
        raise TypeError(f"{field} must be a date, not a datetime")
    if not isinstance(value, date):
        raise TypeError(f"{field} must be a date")
    return value


def _coerce_direction(value: object) -> Direction:
    """Coerce a direction value (``Direction`` or its string value)."""
    if isinstance(value, Direction):
        return value
    if isinstance(value, str):
        try:
            return Direction(value)
        except ValueError:
            raise ValueError(f"invalid direction: {value!r}") from None
    raise TypeError("direction must be a Direction or its string value")


def _coerce_status(value: object) -> TransactionStatus:
    """Coerce a status value (``TransactionStatus`` or its string value)."""
    if isinstance(value, TransactionStatus):
        return value
    if isinstance(value, str):
        try:
            return TransactionStatus(value)
        except ValueError:
            raise ValueError(f"invalid transaction status: {value!r}") from None
    raise TypeError("status must be a TransactionStatus or its string value")


def _validate_transaction_id(value: object) -> str | None:
    """Validate the persistence-assigned identifier (``None`` before save)."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("transaction_id must be a string or None")
    if not value:
        raise ValueError("transaction_id must not be an empty string")
    return value


@dataclass(frozen=True, slots=True)
class Transaction:
    """Immutable, validated transaction entity.

    All values are normalised and validated during construction. Because the
    dataclass is frozen, ordinary attribute assignment cannot silently bypass
    validation.

    ``amount_usdt`` accepts a :class:`decimal.Decimal`, an ``int``, or a
    clean decimal string, and is always stored as a positive ``Decimal``
    with full, unquantised precision. Categories and sources are open-ended
    user data and deliberately not modelled as closed enums.
    """

    direction: Direction
    amount_usdt: Decimal
    category: str
    source: str
    transaction_date: date
    created_at: datetime
    updated_at: datetime
    transaction_id: str | None = None
    comment: str | None = None
    status: TransactionStatus = TransactionStatus.ACTIVE
    deleted_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", _coerce_direction(self.direction))
        object.__setattr__(self, "amount_usdt", normalize_usdt_amount(self.amount_usdt))
        object.__setattr__(self, "category", normalize_required_text(self.category, "category"))
        object.__setattr__(self, "source", normalize_required_text(self.source, "source"))
        object.__setattr__(self, "comment", normalize_optional_text(self.comment, "comment"))
        object.__setattr__(
            self, "transaction_date", require_calendar_date(self.transaction_date, "transaction_date")
        )
        object.__setattr__(self, "created_at", require_aware_datetime(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", require_aware_datetime(self.updated_at, "updated_at"))
        object.__setattr__(self, "transaction_id", _validate_transaction_id(self.transaction_id))

        status = _coerce_status(self.status)
        object.__setattr__(self, "status", status)

        if status is TransactionStatus.ACTIVE:
            if self.deleted_at is not None:
                raise ValueError("an ACTIVE transaction must not have deleted_at")
        else:
            if self.deleted_at is None:
                raise ValueError("a DELETED transaction must have deleted_at")
            object.__setattr__(
                self, "deleted_at", require_aware_datetime(self.deleted_at, "deleted_at")
            )
