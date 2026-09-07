"""Tests for the deterministic transaction-creation core (stage B2).

All tests are deterministic: no network, no filesystem mutation, no
Telegram, no wall clock, no randomness, and no external services.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest

from hermes_finance import (
    Direction,
    ParsedTransactionInput,
    Transaction,
    TransactionStatus,
    create_transaction,
    parse_transaction_input,
)

# Fixed, deterministic reference date/times shared by the tests.
_TX_DATE: date = date(2026, 8, 31)
_CREATED_AT: datetime = datetime(2026, 8, 31, 18, 30, 15, tzinfo=UTC)
_UTC_PLUS_3: timedelta = timedelta(hours=3)


def _parsed(**overrides: Any) -> ParsedTransactionInput:
    """Build a valid ParsedTransactionInput with optional overrides."""
    values: dict[str, Any] = {
        "direction": Direction.EXPENSE,
        "amount_usdt": Decimal(10),
        "category": "Инфраструктура",
        "source": "Хостинг",
        "comment": None,
    }
    values.update(overrides)
    return ParsedTransactionInput(**values)


def create_ok(
    parsed: Any,
    *,
    transaction_date: Any = _TX_DATE,
    created_at: Any = _CREATED_AT,
) -> Transaction:
    """Create a transaction that is expected to be valid."""
    return create_transaction(
        parsed,
        transaction_date=transaction_date,
        created_at=created_at,
    )


class TestValidCreation:
    def test_income_parsed_input_creates_active_transaction(self) -> None:
        parsed = _parsed(
            direction=Direction.INCOME,
            amount_usdt=Decimal(25),
            category="Работа",
            source="Проект A",
        )
        result = create_ok(parsed)
        assert isinstance(result, Transaction)
        assert result.direction is Direction.INCOME
        assert result.status is TransactionStatus.ACTIVE

    def test_expense_parsed_input_creates_active_transaction(self) -> None:
        result = create_ok(_parsed())
        assert isinstance(result, Transaction)
        assert result.direction is Direction.EXPENSE
        assert result.status is TransactionStatus.ACTIVE

    def test_exact_decimal_amount_preserved(self) -> None:
        result = create_ok(_parsed(amount_usdt=Decimal("123.45")))
        assert result.amount_usdt == Decimal("123.45")
        assert result.amount_usdt.as_tuple().exponent == -2

    def test_fine_decimal_precision_preserved(self) -> None:
        amount = Decimal("1.234567890123456789012345789")
        result = create_ok(_parsed(amount_usdt=amount))
        assert result.amount_usdt == amount
        assert result.amount_usdt == Decimal("1.234567890123456789012345789")

    def test_category_preserved(self) -> None:
        result = create_ok(_parsed(category="Сервисы"))
        assert result.category == "Сервисы"

    def test_multi_word_source_preserved(self) -> None:
        result = create_ok(_parsed(source="Крипто Биржа Бета"))
        assert result.source == "Крипто Биржа Бета"

    def test_comment_preserved(self) -> None:
        result = create_ok(_parsed(comment="пополнение резерва"))
        assert result.comment == "пополнение резерва"

    def test_none_comment_preserved(self) -> None:
        result = create_ok(_parsed(comment=None))
        assert result.comment is None

    def test_caller_transaction_date_preserved(self) -> None:
        other_date = date(2025, 1, 2)
        result = create_ok(_parsed(), transaction_date=other_date)
        assert result.transaction_date == other_date
        assert result.transaction_date is other_date

    def test_transaction_date_not_derived_from_created_at(self) -> None:
        # The business date is deliberately different from the creation date:
        # B2 must never derive transaction_date from created_at.
        created = datetime(2026, 9, 1, 3, 0, 0, tzinfo=UTC)
        result = create_ok(_parsed(), transaction_date=date(2026, 8, 31), created_at=created)
        assert result.transaction_date == date(2026, 8, 31)

    def test_caller_created_at_preserved(self) -> None:
        created = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)
        result = create_ok(_parsed(), created_at=created)
        assert result.created_at == created
        assert result.created_at is created

    def test_non_utc_timezone_preserved_exactly(self) -> None:
        tz = timezone(_UTC_PLUS_3)
        created = datetime(2026, 8, 31, 21, 5, 0, tzinfo=tz)
        result = create_ok(_parsed(), created_at=created)
        assert result.created_at.tzinfo is tz
        assert result.created_at.utcoffset() == _UTC_PLUS_3

    def test_updated_at_equals_created_at(self) -> None:
        result = create_ok(_parsed())
        assert result.updated_at == _CREATED_AT
        assert result.updated_at == result.created_at

    def test_transaction_id_is_none(self) -> None:
        assert create_ok(_parsed()).transaction_id is None

    def test_status_is_active(self) -> None:
        assert create_ok(_parsed()).status is TransactionStatus.ACTIVE

    def test_deleted_at_is_none(self) -> None:
        assert create_ok(_parsed()).deleted_at is None

    def test_result_equals_direct_domain_construction(self) -> None:
        # B2 adds no transformation of its own: the created transaction is
        # exactly the domain constructor applied to the same values.
        parsed = _parsed(comment="проверка делегирования")
        result = create_ok(parsed)
        expected = Transaction(
            direction=parsed.direction,
            amount_usdt=parsed.amount_usdt,
            category=parsed.category,
            source=parsed.source,
            comment=parsed.comment,
            transaction_date=_TX_DATE,
            created_at=_CREATED_AT,
            updated_at=_CREATED_AT,
        )
        assert result == expected

    def test_full_pipeline_from_parse_text(self) -> None:
        parsed = parse_transaction_input("-1.5 USDT Инфраструктура Облачный Сервер | замена")
        result = create_ok(parsed)
        assert result.direction is Direction.EXPENSE
        assert result.amount_usdt == Decimal("1.5")
        assert result.category == "Инфраструктура"
        assert result.source == "Облачный Сервер"
        assert result.comment == "замена"
        assert result.transaction_date == _TX_DATE
        assert result.created_at == _CREATED_AT


class TestInputSafety:
    def test_rejects_string_input(self) -> None:
        with pytest.raises(TypeError, match="ParsedTransactionInput"):
            create_ok("-10 Инфраструктура Хостинг")

    def test_rejects_dict_input(self) -> None:
        with pytest.raises(TypeError, match="ParsedTransactionInput"):
            create_ok({"direction": Direction.EXPENSE})

    def test_rejects_transaction_instance(self) -> None:
        # A Transaction is not a ParsedTransactionInput.
        transaction = create_ok(_parsed())
        with pytest.raises(TypeError, match="ParsedTransactionInput"):
            create_ok(transaction)

    def test_rejects_arbitrary_object_without_attribute_leak(self) -> None:
        class Arbitrary:
            pass

        with pytest.raises(TypeError, match="ParsedTransactionInput"):
            create_ok(Arbitrary())

    def test_error_message_names_actual_type(self) -> None:
        with pytest.raises(TypeError, match="'str'"):
            create_ok("not parsed")

    def test_parsed_input_remains_unchanged_after_creation(self) -> None:
        parsed = _parsed(
            direction=Direction.INCOME,
            amount_usdt=Decimal("99.999"),
            category="Работа",
            source="Проект A",
            comment="исходный комментарий",
        )
        snapshot = ParsedTransactionInput(
            direction=parsed.direction,
            amount_usdt=parsed.amount_usdt,
            category=parsed.category,
            source=parsed.source,
            comment=parsed.comment,
        )
        create_ok(parsed, transaction_date=date(2026, 9, 2), created_at=_CREATED_AT)
        assert parsed == snapshot
        assert parsed.direction is Direction.INCOME
        assert parsed.amount_usdt == Decimal("99.999")
        assert parsed.category == "Работа"
        assert parsed.source == "Проект A"
        assert parsed.comment == "исходный комментарий"


class TestTimeSafety:
    def test_naive_created_at_rejected(self) -> None:
        naive = _CREATED_AT.replace(tzinfo=None)
        with pytest.raises(ValueError, match="timezone-aware"):
            create_ok(_parsed(), created_at=naive)

    def test_datetime_as_transaction_date_rejected(self) -> None:
        with pytest.raises(TypeError, match="not a datetime"):
            create_ok(_parsed(), transaction_date=_CREATED_AT)

    def test_invalid_transaction_date_rejected_cleanly(self) -> None:
        with pytest.raises(TypeError, match="must be a date"):
            create_ok(_parsed(), transaction_date="2026-08-31")

    def test_none_created_at_rejected_cleanly(self) -> None:
        with pytest.raises(TypeError, match="must be a datetime"):
            create_ok(_parsed(), created_at=None)

    def test_no_wall_clock_dependency(self) -> None:
        # The created transaction depends only on the caller-supplied
        # date/time: repeated creation with the same inputs is identical,
        # far from any plausible wall-clock "now".
        past = datetime(1970, 1, 1, 0, 0, 0, tzinfo=UTC)
        future = datetime(2999, 12, 31, 23, 59, 59, tzinfo=UTC)
        first = create_ok(_parsed(), transaction_date=date(1970, 1, 1), created_at=past)
        second = create_ok(_parsed(), transaction_date=date(1970, 1, 1), created_at=past)
        assert first == second
        assert first.created_at == past
        late = create_ok(_parsed(), transaction_date=date(2999, 12, 31), created_at=future)
        assert late.created_at == future
        assert late.updated_at == future

    def test_ledger_module_contains_no_wall_clock_calls(self) -> None:
        # Static guarantee: the creation layer never consults the wall clock.
        import inspect

        from hermes_finance import ledger

        source = inspect.getsource(ledger)
        for forbidden in (
            "datetime.now",
            "datetime.today",
            "datetime.utcnow",
            "date.today",
            "time.time",
            "time.monotonic",
        ):
            assert forbidden not in source, forbidden


class TestDomainRegression:
    def test_creation_uses_domain_validation_for_transaction_date(self) -> None:
        # The exact domain error (not a competing ledger-side check) is
        # raised for an invalid transaction_date.
        with pytest.raises(TypeError, match="transaction_date must be a date, not a datetime"):
            create_ok(_parsed(), transaction_date=datetime(2026, 8, 31, tzinfo=UTC))

    def test_creation_uses_domain_validation_for_created_at(self) -> None:
        with pytest.raises(ValueError, match="created_at must be timezone-aware"):
            create_ok(_parsed(), created_at=_CREATED_AT.replace(tzinfo=None))

    def test_domain_invalid_parsed_states_cannot_be_constructed(self) -> None:
        # B1/domain validation already makes invalid ParsedTransactionInput
        # values impossible; creation therefore needs no duplicate checks.
        from hermes_finance.parser import TransactionParseError

        with pytest.raises(TransactionParseError):
            ParsedTransactionInput(
                direction=Direction.EXPENSE,
                amount_usdt=Decimal(-10),
                category="Инфраструктура",
                source="Хостинг",
            )

    def test_created_transaction_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        result = create_ok(_parsed())
        with pytest.raises(FrozenInstanceError):
            result.category = "изменённая"  # type: ignore[misc]
