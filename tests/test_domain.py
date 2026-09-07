"""Tests for the Hermes Finance domain foundation (stage A1).

All tests are deterministic: no network, no filesystem access beyond normal
package import, no wall-clock dependence, and no external services.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest

from hermes_finance import (
    Direction,
    Transaction,
    TransactionStatus,
    normalize_usdt_amount,
)

KYIV = timezone(timedelta(hours=3))

FIXED_DATE = date(2026, 9, 1)
FIXED_CREATED_AT = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
FIXED_UPDATED_AT = datetime(2026, 9, 1, 12, 30, 0, tzinfo=UTC)
FIXED_DELETED_AT = datetime(2026, 9, 1, 13, 0, 0, tzinfo=UTC)


def make_transaction(**overrides: Any) -> Transaction:
    """Build a valid Transaction, applying keyword overrides (possibly invalid)."""
    params: dict[str, Any] = {
        "direction": Direction.EXPENSE,
        "amount_usdt": Decimal(25),
        "category": "groceries",
        "source": "card",
        "transaction_date": FIXED_DATE,
        "created_at": FIXED_CREATED_AT,
        "updated_at": FIXED_UPDATED_AT,
    }
    params.update(overrides)
    return Transaction(**params)


class TestValidTransactions:
    def test_valid_income(self) -> None:
        tx = make_transaction(direction=Direction.INCOME, amount_usdt=Decimal("100.50"))
        assert tx.direction is Direction.INCOME
        assert tx.amount_usdt == Decimal("100.50")
        assert tx.status is TransactionStatus.ACTIVE
        assert tx.deleted_at is None
        assert tx.comment is None
        assert tx.transaction_id is None

    def test_valid_expense(self) -> None:
        tx = make_transaction()
        assert tx.direction is Direction.EXPENSE
        assert tx.amount_usdt == Decimal(25)
        assert tx.category == "groceries"
        assert tx.source == "card"
        assert tx.transaction_date == FIXED_DATE
        assert tx.created_at == FIXED_CREATED_AT
        assert tx.updated_at == FIXED_UPDATED_AT

    def test_direction_and_status_accept_string_values(self) -> None:
        tx = make_transaction(direction="income", status="active")
        assert tx.direction is Direction.INCOME
        assert tx.status is TransactionStatus.ACTIVE

    def test_transaction_id_is_optional_and_kept(self) -> None:
        tx = make_transaction(transaction_id="abc-123")
        assert tx.transaction_id == "abc-123"

    def test_immutable_entity_rejects_attribute_assignment(self) -> None:
        tx = make_transaction()
        with pytest.raises(FrozenInstanceError):
            tx.category = "travel"  # type: ignore


class TestUsdtAmount:
    def test_decimal_exactness(self) -> None:
        tx = make_transaction(amount_usdt=Decimal("123.456789"))
        assert tx.amount_usdt == Decimal("123.456789")
        assert str(tx.amount_usdt) == "123.456789"
        assert tx.amount_usdt.as_tuple().exponent == -6

    def test_fine_usdt_precision_is_preserved(self) -> None:
        for raw in ("25", "0.1", "0.000001", "123.456789"):
            assert normalize_usdt_amount(raw) == Decimal(raw)
        tx = make_transaction(amount_usdt=Decimal("0.000001"))
        assert tx.amount_usdt == Decimal("0.000001")
        # No forced quantisation to 2 decimal places.
        assert tx.amount_usdt != Decimal("0.00")

    def test_integer_normalisation(self) -> None:
        amount: Any = 25
        tx = make_transaction(amount_usdt=amount)
        assert tx.amount_usdt == Decimal(25)
        assert isinstance(tx.amount_usdt, Decimal)
        assert tx.amount_usdt.as_tuple().exponent == 0

    def test_decimal_string_normalisation(self) -> None:
        amount: Any = "0.1"
        tx = make_transaction(amount_usdt=amount)
        assert tx.amount_usdt == Decimal("0.1")
        assert normalize_usdt_amount(" 123.456789 ") == Decimal("123.456789")

    def test_float_rejection(self) -> None:
        amount: Any = 25.5
        with pytest.raises(TypeError, match="float"):
            make_transaction(amount_usdt=amount)
        with pytest.raises(TypeError, match="float"):
            normalize_usdt_amount(0.1)

    def test_bool_rejection(self) -> None:
        amount: Any = True
        with pytest.raises(TypeError, match="bool"):
            make_transaction(amount_usdt=amount)
        with pytest.raises(TypeError, match="bool"):
            normalize_usdt_amount(False)

    def test_zero_rejection(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            make_transaction(amount_usdt=Decimal(0))
        with pytest.raises(ValueError, match="positive"):
            normalize_usdt_amount(0)
        for raw in ("0", "0.00", "0.000000"):
            with pytest.raises(ValueError, match="positive"):
                normalize_usdt_amount(raw)

    def test_negative_amount_rejection(self) -> None:
        amount: Any = -5
        with pytest.raises(ValueError, match="positive"):
            make_transaction(amount_usdt=Decimal(-5))
        with pytest.raises(ValueError, match="positive"):
            normalize_usdt_amount(amount)
        with pytest.raises(ValueError, match="clean decimal"):
            normalize_usdt_amount("-0.1")

    def test_nan_rejection(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            make_transaction(amount_usdt=Decimal("NaN"))
        with pytest.raises(ValueError, match="clean decimal"):
            normalize_usdt_amount("NaN")

    def test_positive_infinity_rejection(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            make_transaction(amount_usdt=Decimal("Infinity"))
        with pytest.raises(ValueError, match="clean decimal"):
            normalize_usdt_amount("Infinity")

    def test_negative_infinity_rejection(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            make_transaction(amount_usdt=Decimal("-Infinity"))
        with pytest.raises(ValueError, match="clean decimal"):
            normalize_usdt_amount("-Infinity")

    def test_unclean_amount_string_rejection(self) -> None:
        for raw in ("12,5", "1e5", "1E5", "abc", "25.", ".5", "1_000", "", "   ", "12.5.1"):
            with pytest.raises(ValueError):
                normalize_usdt_amount(raw)

    def test_unsupported_amount_type_rejection(self) -> None:
        with pytest.raises(TypeError, match="unsupported type"):
            normalize_usdt_amount(None)
        with pytest.raises(TypeError, match="unsupported type"):
            normalize_usdt_amount([Decimal(1)])


class TestTextNormalisation:
    def test_empty_category_rejection(self) -> None:
        with pytest.raises(ValueError, match="category"):
            make_transaction(category="")
        with pytest.raises(ValueError, match="category"):
            make_transaction(category="   ")

    def test_empty_source_rejection(self) -> None:
        with pytest.raises(ValueError, match="source"):
            make_transaction(source="")
        with pytest.raises(ValueError, match="source"):
            make_transaction(source="\t \n")

    def test_category_and_source_trimming(self) -> None:
        tx = make_transaction(category="  groceries  ", source="\tcard\n")
        assert tx.category == "groceries"
        assert tx.source == "card"

    def test_blank_comment_becomes_none(self) -> None:
        assert make_transaction(comment="   ").comment is None
        assert make_transaction(comment=None).comment is None
        assert make_transaction(comment=" weekly shop ").comment == "weekly shop"

    def test_non_string_text_rejection(self) -> None:
        with pytest.raises(TypeError, match="category"):
            make_transaction(category=123)
        with pytest.raises(TypeError, match="source"):
            make_transaction(source=None)
        with pytest.raises(TypeError, match="comment"):
            make_transaction(comment=42)

    def test_empty_transaction_id_rejection(self) -> None:
        with pytest.raises(ValueError, match="transaction_id"):
            make_transaction(transaction_id="")


class TestTimestamps:
    def test_timezone_aware_timestamps_accepted(self) -> None:
        created = datetime(2026, 9, 1, 15, 0, 0, tzinfo=KYIV)
        updated = datetime(2026, 9, 1, 16, 0, 0, tzinfo=KYIV)
        tx = make_transaction(created_at=created, updated_at=updated)
        assert tx.created_at == created
        assert tx.updated_at == updated

    def test_naive_created_at_rejected(self) -> None:
        with pytest.raises(ValueError, match="created_at"):
            make_transaction(created_at=FIXED_CREATED_AT.replace(tzinfo=None))

    def test_naive_updated_at_rejected(self) -> None:
        with pytest.raises(ValueError, match="updated_at"):
            make_transaction(updated_at=FIXED_UPDATED_AT.replace(tzinfo=None))

    def test_naive_deleted_at_rejected(self) -> None:
        with pytest.raises(ValueError, match="deleted_at"):
            make_transaction(
                status=TransactionStatus.DELETED,
                deleted_at=FIXED_DELETED_AT.replace(tzinfo=None),
            )

    def test_non_datetime_timestamp_rejection(self) -> None:
        with pytest.raises(TypeError, match="created_at"):
            make_transaction(created_at="2026-09-01T12:00:00+00:00")

    def test_datetime_transaction_date_rejected(self) -> None:
        with pytest.raises(TypeError, match="transaction_date"):
            make_transaction(transaction_date=datetime(2026, 9, 1, tzinfo=UTC))

    def test_non_date_transaction_date_rejected(self) -> None:
        with pytest.raises(TypeError, match="transaction_date"):
            make_transaction(transaction_date="2026-09-01")


class TestLifecycle:
    def test_active_with_deleted_at_rejected(self) -> None:
        with pytest.raises(ValueError, match="ACTIVE"):
            make_transaction(deleted_at=FIXED_DELETED_AT)

    def test_deleted_without_deleted_at_rejected(self) -> None:
        with pytest.raises(ValueError, match="DELETED"):
            make_transaction(status=TransactionStatus.DELETED)

    def test_valid_deleted_transaction(self) -> None:
        tx = make_transaction(
            status=TransactionStatus.DELETED,
            deleted_at=FIXED_DELETED_AT,
        )
        assert tx.status is TransactionStatus.DELETED
        assert tx.deleted_at == FIXED_DELETED_AT

    def test_deleted_accepts_string_status_value(self) -> None:
        tx = make_transaction(status="deleted", deleted_at=FIXED_DELETED_AT)
        assert tx.status is TransactionStatus.DELETED

    def test_invalid_direction_value_rejected(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            make_transaction(direction="transfer")

    def test_invalid_status_value_rejected(self) -> None:
        with pytest.raises(ValueError, match="status"):
            make_transaction(status="pending")

    def test_invalid_direction_type_rejected(self) -> None:
        amount: Any = 1
        with pytest.raises(TypeError, match="direction"):
            make_transaction(direction=amount)
