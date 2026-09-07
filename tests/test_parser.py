"""Tests for the deterministic fast-entry input parser (stage B1).

All tests are deterministic: no network, no filesystem mutation, no
Telegram, no wall clock, no randomness, and no external services.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal
from typing import Any

import pytest

from hermes_finance import (
    Direction,
    ParsedTransactionInput,
    TransactionParseError,
    parse_transaction_input,
)


def parse_ok(text: str) -> ParsedTransactionInput:
    """Parse text that is expected to be valid."""
    return parse_transaction_input(text)


def parse_err(text: Any) -> TransactionParseError:
    """Parse text that is expected to be invalid; return the exception."""
    with pytest.raises(TransactionParseError) as excinfo:
        parse_transaction_input(text)
    return excinfo.value


class TestValidInput:
    def test_income_sign(self) -> None:
        result = parse_ok("+25 Работа Проект A")
        assert result.direction is Direction.INCOME
        assert result.amount_usdt == Decimal(25)
        assert result.category == "Работа"
        assert result.source == "Проект A"
        assert result.comment is None

    def test_expense_sign(self) -> None:
        result = parse_ok("-10 Инфраструктура Хостинг")
        assert result.direction is Direction.EXPENSE
        assert result.amount_usdt == Decimal(10)
        assert result.category == "Инфраструктура"
        assert result.source == "Хостинг"

    def test_integer_amount(self) -> None:
        assert parse_ok("+7 Работа Bot").amount_usdt == Decimal(7)

    def test_decimal_amount(self) -> None:
        assert parse_ok("+123.456789 USDT Сервисы Подписка").amount_usdt == Decimal("123.456789")

    def test_fine_decimal_precision_is_preserved_exactly(self) -> None:
        result = parse_ok("+0.000001 Работа Bot")
        assert result.amount_usdt == Decimal("0.000001")
        assert result.amount_usdt.as_tuple().exponent == -6
        assert result.amount_usdt.is_finite()

    def test_amount_precision_is_not_quantised(self) -> None:
        result = parse_ok("+1.234567890123456789012345789 Работа Bot")
        assert result.amount_usdt == Decimal("1.234567890123456789012345789")

    def test_usdt_attached_to_amount(self) -> None:
        result = parse_ok("-1USDT Инфраструктура Сервер A")
        assert result.amount_usdt == Decimal(1)
        assert result.category == "Инфраструктура"
        assert result.source == "Сервер A"
        assert result.comment is None

    def test_usdt_separated_from_amount(self) -> None:
        result = parse_ok("+25 USDT Сервисы Подписка")
        assert result.amount_usdt == Decimal(25)
        assert result.category == "Сервисы"
        assert result.source == "Подписка"

    def test_usdt_ascii_case_insensitive(self) -> None:
        for token in ("USDT", "usdt", "UsDt"):
            result = parse_ok(f"+25 {token} Сервисы Подписка")
            assert result.category == "Сервисы"
            assert result.source == "Подписка"
        result = parse_ok("+25UsDt Сервисы Подписка")
        assert result.category == "Сервисы"
        assert result.source == "Подписка"

    def test_usdt_ascii_only_case_forms_all_accept(self) -> None:
        # Regression (QA B1): USDT recognition is ASCII case-insensitive
        # ONLY. Every ASCII spelling of U/S/D/T must be accepted in both
        # the separated and the glued form.
        for token in ("USDT", "usdt", "UsDt", "uSdT"):
            for separated in (True, False):
                text = f"+25 {token} Cat Src" if separated else f"+25{token} Cat Src"
                result = parse_ok(text)
                assert result.category == "Cat"
                assert result.source == "Src"
                assert result.amount_usdt == Decimal(25)

    def test_usdt_non_ascii_long_s_lookalike_separated_is_rejected(self) -> None:
        # "ſ" is U+017F LATIN SMALL LETTER LONG S; str.upper() maps it to
        # "S", so "uſdt".upper() == "USDT". It must NOT be accepted as the
        # USDT currency token, and must fail through TransactionParseError.
        error = parse_err("+25 uſdt Cat Src")
        assert type(error) is TransactionParseError

    def test_usdt_non_ascii_long_s_lookalike_glued_is_rejected(self) -> None:
        error = parse_err("+25uſdt Cat Src")
        assert type(error) is TransactionParseError

    def test_usdt_non_ascii_lookalike_uppercase_variant_is_rejected(self) -> None:
        # Additional Unicode case-conversion probe: "UſDt".upper() == "USDT"
        # through the long-s mapping, but it is not an ASCII spelling.
        error = parse_err("+25 UſDt Cat Src")
        assert type(error) is TransactionParseError

    def test_non_ascii_usdt_lookalike_error_message_mentions_token(self) -> None:
        error = parse_err("+25uſdt Cat Src")
        assert "uſdt" in str(error)

    def test_no_usdt_token(self) -> None:
        result = parse_ok("-12 Инфраструктура Хостинг")
        assert result.category == "Инфраструктура"
        assert result.source == "Хостинг"

    def test_cyrillic_category_and_source(self) -> None:
        result = parse_ok("-12 Инфраструктура Хостинг Германия")
        assert result.category == "Инфраструктура"
        assert result.source == "Хостинг Германия"

    def test_multi_word_source(self) -> None:
        result = parse_ok("-12 Инфраструктура Хостинг Германия")
        assert result.source == "Хостинг Германия"

    def test_leading_and_trailing_input_whitespace(self) -> None:
        result = parse_ok("   \t +25 Работа Проект A \n ")
        assert result.amount_usdt == Decimal(25)
        assert result.category == "Работа"
        assert result.source == "Проект A"

    def test_multiple_structural_spaces_collapse_inside_source(self) -> None:
        # Documented deterministic normalisation: runs of structural
        # whitespace between tokens collapse to single spaces.
        result = parse_ok("-12   Инфраструктура   Хостинг   Германия")
        assert result.category == "Инфраструктура"
        assert result.source == "Хостинг Германия"

    def test_tabs_and_newlines_as_structural_separators(self) -> None:
        result = parse_ok("-12\tИнфраструктура\n\tХостинг Германия")
        assert result.category == "Инфраструктура"
        assert result.source == "Хостинг Германия"

    def test_comment(self) -> None:
        result = parse_ok("-10 Инфраструктура Хостинг | основной аккаунт")
        assert result.comment == "основной аккаунт"

    def test_blank_comment_normalises_to_none(self) -> None:
        result = parse_ok("-10 Инфраструктура Хостинг |")
        assert result.comment is None

    def test_blank_comment_with_whitespace_normalises_to_none(self) -> None:
        result = parse_ok("-10 Инфраструктура Хостинг |   ")
        assert result.comment is None

    def test_comment_preserves_additional_pipe_characters(self) -> None:
        result = parse_ok("-10 Инфраструктура Хостинг | paid | monthly")
        assert result.comment == "paid | monthly"

    def test_comment_internal_spacing_is_preserved(self) -> None:
        result = parse_ok("-10 Инфраструктура Хостинг | основной   аккаунт")
        assert result.comment == "основной   аккаунт"

    def test_comment_without_space_around_delimiter(self) -> None:
        result = parse_ok("-10 Инфраструктура Хостинг|основной аккаунт")
        assert result.category == "Инфраструктура"
        assert result.source == "Хостинг"
        assert result.comment == "основной аккаунт"

    def test_exact_decimal_string_preserved(self) -> None:
        result = parse_ok("+123.456789 USDT Сервисы Подписка")
        assert str(result.amount_usdt) == "123.456789"


class TestInvalidInput:
    def test_non_string_input(self) -> None:
        error = parse_err(123)
        assert isinstance(error, ValueError)

    def test_non_string_none_input(self) -> None:
        parse_err(None)

    def test_blank_input(self) -> None:
        parse_err("   \t\n ")

    def test_missing_sign(self) -> None:
        parse_err("25 Работа Проект A")

    def test_plus_without_amount(self) -> None:
        parse_err("+")

    def test_minus_without_amount(self) -> None:
        parse_err("-")

    def test_whitespace_between_sign_and_amount(self) -> None:
        parse_err("+ 25 Работа Проект A")

    def test_whitespace_between_sign_and_amount_minus(self) -> None:
        parse_err("- 10 Инфраструктура Хостинг")

    def test_zero_amount(self) -> None:
        parse_err("+0 Работа Проект A")

    def test_zero_decimal_amount(self) -> None:
        parse_err("+0.000 Работа Проект A")

    def test_malformed_decimal_leading_dot(self) -> None:
        parse_err("+.5 Работа Проект A")

    def test_malformed_decimal_trailing_dot(self) -> None:
        parse_err("+5. Работа Проект A")

    def test_malformed_decimal_double_dot(self) -> None:
        parse_err("+1.2.3 Работа Проект A")

    def test_exponent_notation(self) -> None:
        parse_err("+1e3 Работа Проект A")

    def test_exponent_notation_uppercase(self) -> None:
        parse_err("+1E3 Работа Проект A")

    def test_comma_decimal(self) -> None:
        parse_err("+1,5 Работа Проект A")

    def test_currency_symbol(self) -> None:
        parse_err("+$10 Работа Проект A")

    def test_unsupported_usd_currency_token(self) -> None:
        # Must NOT be reinterpreted as category="USD".
        error = parse_err("+25 USD Работа Проект A")
        assert "USD" in str(error)

    def test_unsupported_currency_like_uppercase_token(self) -> None:
        parse_err("+25 EUR Работа Проект A")

    def test_unsupported_currency_like_token_four_letters(self) -> None:
        parse_err("+25 USDC Работа Проект A")

    def test_unsupported_currency_token_glued_to_amount(self) -> None:
        parse_err("-10USD Инфраструктура Хостинг")

    def test_only_signed_amount(self) -> None:
        parse_err("+25")

    def test_amount_and_usdt_only(self) -> None:
        parse_err("+25 USDT")

    def test_amount_and_glued_usdt_only(self) -> None:
        parse_err("+25USDT")

    def test_missing_category_amount_and_source_only(self) -> None:
        # "-10 Инфраструктура" has category but no source; the missing-source
        # variant is covered separately. Here: amount + single token.
        parse_err("-10")

    def test_amount_category_without_source(self) -> None:
        parse_err("-10 Инфраструктура")

    def test_missing_category_with_comment_delimiter(self) -> None:
        parse_err("+25 | комментарий")

    def test_malformed_glued_suffix_after_amount(self) -> None:
        parse_err("-10Инфраструктура Хостинг")

    def test_malformed_glued_digit_text_after_amount(self) -> None:
        parse_err("+25x Работа Проект A")

    def test_unsupported_text_before_sign(self) -> None:
        parse_err("extra +25 Работа Проект A")

    def test_internal_sign_is_not_a_leading_sign(self) -> None:
        parse_err("25 +10 Работа Проект A")


class TestParsedTransactionInputContract:
    def test_result_is_immutable(self) -> None:
        result = parse_ok("+25 Работа Проект A")
        with pytest.raises(FrozenInstanceError):
            result.category = "Другое"  # type: ignore[misc]

    def test_result_type(self) -> None:
        result = parse_ok("+25 Работа Проект A")
        assert isinstance(result, ParsedTransactionInput)

    def test_comment_defaults_to_none(self) -> None:
        result = ParsedTransactionInput(
            direction=Direction.INCOME,
            amount_usdt=Decimal(5),
            category="Cat",
            source="Src",
        )
        assert result.comment is None

    def test_direct_construction_validates_positive_amount(self) -> None:
        with pytest.raises(TransactionParseError):
            ParsedTransactionInput(
                direction=Direction.INCOME,
                amount_usdt=Decimal(0),
                category="Cat",
                source="Src",
            )

    def test_direct_construction_validates_empty_category(self) -> None:
        with pytest.raises(TransactionParseError):
            ParsedTransactionInput(
                direction=Direction.INCOME,
                amount_usdt=Decimal(5),
                category="  ",
                source="Src",
            )

    def test_direct_construction_rejects_invalid_direction(self) -> None:
        with pytest.raises(TransactionParseError):
            ParsedTransactionInput(
                direction="income",  # type: ignore[arg-type]
                amount_usdt=Decimal(5),
                category="Cat",
                source="Src",
            )

    def test_error_is_value_error_subclass(self) -> None:
        assert issubclass(TransactionParseError, ValueError)

    def test_parse_error_is_raised_not_leaked_internals(self) -> None:
        # The public contract raises TransactionParseError, never bare
        # IndexError/AttributeError/Decimal/regex internals.
        error = parse_err("+")
        assert type(error) is TransactionParseError
