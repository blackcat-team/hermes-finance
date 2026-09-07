"""Tests for the deterministic plain-text report renderer (stage E2).

All tests are deterministic: no network, no wall-clock time, no
randomness, no environment access. Filesystem mutation happens only
inside in-memory SQLite databases (``:memory:``), and only in the
end-to-end tests that exercise DB -> E1 -> E2.

The renderer is a pure presentation-only projection: every exact-string
test proves that stored report values are displayed verbatim, never
recalculated, never reordered, and never reformatted through floats,
rounding, quantisation, or the ambient decimal context.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, date, datetime
from decimal import Decimal, getcontext, localcontext
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

import pytest

import hermes_finance
import hermes_finance.rendering as rendering_module
from hermes_finance import (
    CategoryReport,
    Direction,
    FinanceConfig,
    MonthlyFinanceReport,
    SourceReport,
    Transaction,
    TransactionStatus,
    build_monthly_report,
    edit_transaction,
    list_recent_transactions,
    list_transactions_by_date,
    list_transactions_by_month,
    open_database,
    persist_transaction,
    render_monthly_report,
    render_recent_transactions,
    render_transactions_by_date,
    render_transactions_by_month,
    soft_delete_transaction,
)
from hermes_finance.provenance import TelegramMessageRef

RENDERING_SOURCE: str = Path(rendering_module.__file__).read_text(encoding="utf-8")

FIXED_TZ = UTC
FIXED_CREATED_AT = datetime(2026, 9, 1, 10, 0, 0, tzinfo=FIXED_TZ)
FIXED_UPDATED_AT = datetime(2026, 9, 1, 11, 30, 0, tzinfo=FIXED_TZ)
FIXED_PROCESSED_AT = datetime(2026, 9, 1, 12, 15, 0, tzinfo=FIXED_TZ)
FIXED_DELETED_AT = datetime(2026, 9, 1, 13, 0, 0, tzinfo=FIXED_TZ)

ZERO: Final[Decimal] = Decimal(0)

MONTH_NAMES: Final[dict[int, str]] = {
    1: "Январь",
    2: "Февраль",
    3: "Март",
    4: "Апрель",
    5: "Май",
    6: "Июнь",
    7: "Июль",
    8: "Август",
    9: "Сентябрь",
    10: "Октябрь",
    11: "Ноябрь",
    12: "Декабрь",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def conn() -> Iterator[sqlite3.Connection]:
    """A fresh migrated in-memory database connection."""
    connection = open_database(
        FinanceConfig(database_path=Path(":memory:"), business_timezone=FIXED_TZ)
    )
    try:
        yield connection
    finally:
        connection.close()


def make_ref(*, message_id: int, update_id: int) -> TelegramMessageRef:
    """A valid Telegram message provenance reference."""
    return TelegramMessageRef(
        chat_id=-100200,
        message_thread_id=7,
        message_id=message_id,
        update_id=update_id,
    )


def make_transaction(**overrides: Any) -> Transaction:
    """A valid unpersisted Transaction with field overrides."""
    values: dict[str, Any] = {
        "direction": Direction.INCOME,
        "amount_usdt": Decimal(25),
        "category": "Работа",
        "source": "Проект A",
        "comment": None,
        "transaction_date": date(2026, 8, 1),
        "created_at": FIXED_CREATED_AT,
        "updated_at": FIXED_UPDATED_AT,
        "status": TransactionStatus.ACTIVE,
        "deleted_at": None,
    }
    values.update(overrides)
    return Transaction(**values)


_next_message_id = 0


def persist_row(
    connection: sqlite3.Connection,
    *,
    transaction_date: date,
    **transaction_overrides: Any,
) -> Transaction:
    """Persist one ACTIVE transaction with a unique provenance identity."""
    global _next_message_id
    _next_message_id += 1
    message_id = _next_message_id
    return persist_transaction(
        connection,
        make_transaction(transaction_date=transaction_date, **transaction_overrides),
        make_ref(message_id=message_id, update_id=10_000 + message_id),
        processed_at=FIXED_PROCESSED_AT,
    )


def persist_canonical_august(connection: sqlite3.Connection) -> None:
    """Persist the canonical six-transaction E1 example month (2026-08).

    income:
    +25 Работа Проект A, +30 Работа Проект B, +5 Сервисы Подписка
    expense:
    -10 Инфраструктура Хостинг, -1 Инфраструктура Сервер A, -10 Инфраструктура Сервер B
    """
    persist_row(
        connection,
        transaction_date=date(2026, 8, 1),
        amount_usdt=Decimal(25),
        category="Работа",
        source="Проект A",
    )
    persist_row(
        connection,
        transaction_date=date(2026, 8, 9),
        amount_usdt=Decimal(30),
        category="Работа",
        source="Проект B",
    )
    persist_row(
        connection,
        transaction_date=date(2026, 8, 7),
        amount_usdt=Decimal(5),
        category="Сервисы",
        source="Подписка",
    )
    persist_row(
        connection,
        transaction_date=date(2026, 8, 11),
        direction=Direction.EXPENSE,
        amount_usdt=Decimal(10),
        category="Инфраструктура",
        source="Хостинг",
    )
    persist_row(
        connection,
        transaction_date=date(2026, 8, 5),
        direction=Direction.EXPENSE,
        amount_usdt=Decimal(1),
        category="Инфраструктура",
        source="Сервер A",
    )
    persist_row(
        connection,
        transaction_date=date(2026, 8, 3),
        direction=Direction.EXPENSE,
        amount_usdt=Decimal(10),
        category="Инфраструктура",
        source="Сервер B",
    )


def build_report(
    *,
    year: int = 2026,
    month: int = 8,
    income: Decimal | int = Decimal(0),
    expense: Decimal | int = Decimal(0),
    net: Decimal | int = Decimal(0),
    count: int = 0,
    categories: tuple[CategoryReport, ...] = (),
) -> MonthlyFinanceReport:
    """A directly constructed report for renderer-level tests."""
    return MonthlyFinanceReport(
        year=year,
        month=month,
        income_usdt=Decimal(income),
        expense_usdt=Decimal(expense),
        net_usdt=Decimal(net),
        transaction_count=count,
        categories=categories,
    )


def build_source(
    source: str,
    *,
    income: Decimal | int = Decimal(0),
    expense: Decimal | int = Decimal(0),
    net: Decimal | int = Decimal(0),
    count: int = 0,
) -> SourceReport:
    """A directly constructed source report."""
    return SourceReport(
        source=source,
        income_usdt=Decimal(income),
        expense_usdt=Decimal(expense),
        net_usdt=Decimal(net),
        transaction_count=count,
    )


def build_category(
    category: str,
    *,
    income: Decimal | int = Decimal(0),
    expense: Decimal | int = Decimal(0),
    net: Decimal | int = Decimal(0),
    count: int = 0,
    sources: tuple[SourceReport, ...] = (),
) -> CategoryReport:
    """A directly constructed category report."""
    return CategoryReport(
        category=category,
        income_usdt=Decimal(income),
        expense_usdt=Decimal(expense),
        net_usdt=Decimal(net),
        transaction_count=count,
        sources=sources,
    )


def render_lines(rendered: str) -> list[str]:
    """The rendered report split into physical lines."""
    return rendered.split("\n")


# ---------------------------------------------------------------------------
# exact canonical end-to-end report: DB -> E1 -> E2
# ---------------------------------------------------------------------------

CANONICAL_AUGUST: Final[str] = (
    "💰 FINANCE | АВГУСТ 2026\n"
    "\n"
    "📈 Доход: 60 USDT\n"
    "📉 Расход: 21 USDT\n"
    "⚖️ Итог: +39 USDT\n"
    "🧾 Операций: 6\n"
    "\n"
    "📂 Инфраструктура\n"
    "Доход: 0 USDT\n"
    "Расход: 21 USDT\n"
    "Итог: -21 USDT\n"
    "Операций: 3\n"
    "\n"
    "🔹 Источники\n"
    "• Сервер A\n"
    "  доход 0 · расход 1 · итог -1 USDT · операций 1\n"
    "• Сервер B\n"
    "  доход 0 · расход 10 · итог -10 USDT · операций 1\n"
    "• Хостинг\n"
    "  доход 0 · расход 10 · итог -10 USDT · операций 1\n"
    "\n"
    "📂 Работа\n"
    "Доход: 55 USDT\n"
    "Расход: 0 USDT\n"
    "Итог: +55 USDT\n"
    "Операций: 2\n"
    "\n"
    "🔹 Источники\n"
    "• Проект A\n"
    "  доход 25 · расход 0 · итог +25 USDT · операций 1\n"
    "• Проект B\n"
    "  доход 30 · расход 0 · итог +30 USDT · операций 1\n"
    "\n"
    "📂 Сервисы\n"
    "Доход: 5 USDT\n"
    "Расход: 0 USDT\n"
    "Итог: +5 USDT\n"
    "Операций: 1\n"
    "\n"
    "🔹 Источники\n"
    "• Подписка\n"
    "  доход 5 · расход 0 · итог +5 USDT · операций 1"
)


def test_canonical_six_transaction_report_exact_string(conn: sqlite3.Connection) -> None:
    """End-to-end DB -> E1 -> E2 with exact full-string equality.

    The canonical August 2026 month (income 60, expense 21, net 39,
    count 6) must render with exact whitespace, exact E1 ordering
    (categories and sources), and no final newline.
    """
    persist_canonical_august(conn)
    report = build_monthly_report(conn, year=2026, month=8)

    rendered = render_monthly_report(report)

    assert rendered == CANONICAL_AUGUST
    # Output structure: no final newline, no trailing whitespace.
    assert not rendered.endswith("\n")
    assert all(line == line.rstrip() for line in render_lines(rendered))
    # No tabs anywhere.
    assert "\t" not in rendered
    # The accepted BlackCat identity header is present; the old technical
    # header and the generic category boilerplate are gone.
    assert render_lines(rendered)[0] == "💰 FINANCE | АВГУСТ 2026"
    assert "📊 Финансы •" not in rendered
    assert "Категории:" not in rendered


def test_canonical_report_blank_line_structure(conn: sqlite3.Connection) -> None:
    """One blank line after the metrics, before 🔹, and between categories."""
    persist_canonical_august(conn)
    rendered = render_monthly_report(build_monthly_report(conn, year=2026, month=8))
    lines = render_lines(rendered)

    assert lines[0] == "💰 FINANCE | АВГУСТ 2026"
    assert lines[1] == ""
    assert lines[2] == "📈 Доход: 60 USDT"
    assert lines[5] == "🧾 Операций: 6"
    assert lines[6] == ""
    assert lines[7] == "📂 Инфраструктура"
    markers = [index for index, line in enumerate(lines) if line.startswith("📂 ")]
    assert markers == [7, 21, 33]
    for marker in markers:
        assert lines[marker - 1] == ""  # exactly one blank line before each category
    # One blank line before each 🔹 Источники marker.
    source_markers = [index for index, line in enumerate(lines) if line == "🔹 Источники"]
    assert source_markers == [13, 27, 39]
    for marker in source_markers:
        assert lines[marker - 1] == ""
    # No two consecutive blank lines anywhere.
    for previous, current in pairwise(lines):
        assert not (previous == "" and current == "")


# ---------------------------------------------------------------------------
# empty month
# ---------------------------------------------------------------------------


def test_empty_month_exact_string() -> None:
    """An empty report renders the empty-month format exactly."""
    report = build_report(year=2026, month=5)

    rendered = render_monthly_report(report)

    assert rendered == (
        "💰 FINANCE | МАЙ 2026\n"
        "\n"
        "📈 Доход: 0 USDT\n"
        "📉 Расход: 0 USDT\n"
        "⚖️ Итог: 0 USDT\n"
        "🧾 Операций: 0\n"
        "\n"
        "📭 Операций за месяц нет."
    )
    assert "Категории:" not in rendered
    assert rendered != ""


def test_empty_september_2026_blackcat_card_exact_string() -> None:
    """The canonical empty September 2026 BlackCat card, pinned exactly.

    Explicitly asserts the accepted identity header, the absence of the
    old technical header and the generic category boilerplate, the
    emoji-led zero summary, the empty-month marker, no trailing
    whitespace, no final newline, and deterministic repeat rendering.
    """
    report = build_report(year=2026, month=9)

    rendered = render_monthly_report(report)

    assert rendered == (
        "💰 FINANCE | СЕНТЯБРЬ 2026\n"
        "\n"
        "📈 Доход: 0 USDT\n"
        "📉 Расход: 0 USDT\n"
        "⚖️ Итог: 0 USDT\n"
        "🧾 Операций: 0\n"
        "\n"
        "📭 Операций за месяц нет."
    )
    assert render_lines(rendered)[0] == "💰 FINANCE | СЕНТЯБРЬ 2026"
    assert "📊 Финансы •" not in rendered
    assert "Категории:" not in rendered
    assert "📭 Операций за месяц нет." in rendered
    assert not rendered.endswith("\n")
    assert all(line == line.rstrip() for line in render_lines(rendered))
    # Deterministic repeat render: a pure projection renders identically.
    assert render_monthly_report(report) == rendered


def test_empty_month_from_empty_database(conn: sqlite3.Connection) -> None:
    """E1 empty month flows through E2 unchanged."""
    report = build_monthly_report(conn, year=2026, month=5)

    rendered = render_monthly_report(report)

    assert "Операций: 0" in rendered
    assert "📭 Операций за месяц нет." in rendered
    assert "Категории:" not in rendered


@pytest.mark.parametrize(
    ("month", "name"),
    sorted(MONTH_NAMES.items()),
)
def test_all_russian_month_names(month: int, name: str) -> None:
    """Every calendar month renders its hard-coded uppercase Russian name."""
    report = build_report(year=2026, month=month)
    rendered = render_monthly_report(report)
    assert render_lines(rendered)[0] == f"💰 FINANCE | {name.upper()} 2026"


# ---------------------------------------------------------------------------
# sign policy
# ---------------------------------------------------------------------------


def test_negative_net_exact_string() -> None:
    """A negative net renders exactly one minus, at all three levels."""
    report = build_report(
        year=2026,
        month=7,
        income=5,
        expense=16,
        net=-11,
        count=2,
        categories=(
            build_category(
                "Инфраструктура",
                income=0,
                expense=16,
                net=-16,
                count=1,
                sources=(build_source("Сервер A", expense=16, net=-16, count=1),),
            ),
            build_category(
                "Работа",
                income=5,
                expense=0,
                net=5,
                count=1,
                sources=(build_source("Проект A", income=5, net=5, count=1),),
            ),
        ),
    )

    rendered = render_monthly_report(report)

    assert "Итог: -11 USDT" in rendered
    assert "Итог: --11 USDT" not in rendered
    assert "Итог: -16 USDT" in rendered
    assert "итог -16 USDT" in rendered
    assert "Итог: +5 USDT" in rendered


def test_zero_net_exact_string() -> None:
    """An exact zero net renders as 0, never +0, -0, or 0.00."""
    report = build_report(
        year=2026,
        month=6,
        income=21,
        expense=21,
        net=0,
        count=2,
        categories=(
            build_category(
                "Работа",
                income=21,
                expense=21,
                net=0,
                count=2,
                sources=(build_source("Проект A", income=21, expense=21, net=0, count=2),),
            ),
        ),
    )

    rendered = render_monthly_report(report)

    assert "Итог: 0 USDT" in rendered
    assert "Итог: +0" not in rendered
    assert "Итог: -0" not in rendered
    assert "0.00" not in rendered
    assert "итог 0 USDT" in rendered


def test_income_expense_render_without_explicit_plus() -> None:
    """Positive magnitudes never carry a leading +."""
    report = build_report(year=2026, month=8, income=60, expense=21, net=39, count=1)
    rendered = render_monthly_report(report)
    assert "Доход: 60 USDT" in rendered
    assert "Расход: 21 USDT" in rendered
    assert "Доход: +60" not in rendered
    assert "Итог: +39 USDT" in rendered


# ---------------------------------------------------------------------------
# decimal display precision
# ---------------------------------------------------------------------------


def test_tiny_and_trailing_zero_decimals() -> None:
    """Trailing fractional zeros are removed; tiny values keep all digits."""
    report = build_report(
        year=2026,
        month=9,
        income=Decimal("123.4500"),
        expense=Decimal("0.000002"),
        net=Decimal("123.449998"),
        count=2,
        categories=(
            build_category(
                "Работа",
                income=Decimal("123.4500"),
                expense=Decimal("0.000002"),
                net=Decimal("123.449998"),
                count=2,
                sources=(
                    build_source(
                        "Проект B",
                        income=Decimal("123.4500"),
                        expense=Decimal("0.000002"),
                        net=Decimal("123.449998"),
                        count=2,
                    ),
                ),
            ),
        ),
    )

    rendered = render_monthly_report(report)

    assert "Доход: 123.45 USDT" in rendered
    assert "Расход: 0.000002 USDT" in rendered
    assert "Итог: +123.449998 USDT" in rendered
    assert "123.4500" not in rendered
    assert "123.45 " in rendered  # not a truncated "123.4"
    assert "0.000001" not in rendered  # not rounded


def test_integral_decimal_renders_without_point() -> None:
    """Decimal("60.0") renders as 60; Decimal("60") also renders as 60."""
    report = build_report(
        year=2026,
        month=8,
        income=Decimal("60.0"),
        expense=Decimal("21.00"),
        net=Decimal("39.0"),
        count=1,
    )
    rendered = render_monthly_report(report)
    assert "Доход: 60 USDT" in rendered
    assert "Расход: 21 USDT" in rendered
    assert "Итог: +39 USDT" in rendered
    assert ".0" not in rendered
    assert ".00" not in rendered


def test_hundred_digit_integers_exact() -> None:
    """100+ significant integer digits survive exactly, no scientific notation."""
    big_income = Decimal("9" * 120)
    big_expense = Decimal("1" + "0" * 119)
    net = Decimal("8" * 120)
    report = build_report(
        year=2026,
        month=8,
        income=big_income,
        expense=big_expense,
        net=net,
        count=1,
        categories=(
            build_category(
                "Работа",
                income=big_income,
                expense=big_expense,
                net=net,
                count=1,
                sources=(build_source("Проект B", income=big_income, net=net, count=1),),
            ),
        ),
    )

    rendered = render_monthly_report(report)

    assert f"Доход: {'9' * 120} USDT" in rendered
    assert f"Расход: {'1' + '0' * 119} USDT" in rendered
    assert f"Итог: +{'8' * 120} USDT" in rendered
    assert "E+" not in rendered
    assert "E-" not in rendered


def test_large_integer_plus_tiny_fraction_exact() -> None:
    """A value produced by exact E1 aggregation renders with both scales."""
    huge = Decimal("999999999999999999.999999999999999999")
    report = build_report(year=2026, month=8, income=huge, expense=0, net=huge, count=1)
    rendered = render_monthly_report(report)
    assert "Доход: 999999999999999999.999999999999999999 USDT" in rendered
    assert "Итог: +999999999999999999.999999999999999999 USDT" in rendered


def test_positive_exponent_decimal_renders_fixed_point() -> None:
    """A Decimal stored with a positive exponent renders expanded."""
    value = Decimal((0, (1, 2, 3), 3))  # exactly 123000, stored as 123E+3
    assert str(value) == "1.23E+5"
    report = build_report(year=2026, month=8, income=value, expense=0, net=value, count=1)
    rendered = render_monthly_report(report)
    assert "Доход: 123000 USDT" in rendered
    assert "Итог: +123000 USDT" in rendered


# ---------------------------------------------------------------------------
# hostile ambient decimal context
# ---------------------------------------------------------------------------


def test_hostile_context_precision_exact_and_unchanged() -> None:
    """Under ambient prec=2, 100+ digit values render exactly, context untouched."""
    digits = "3" * 110
    value = Decimal(digits)
    report = build_report(
        year=2026,
        month=8,
        income=value,
        expense=Decimal(1),
        net=Decimal("3" * 110),
        count=1,
    )
    default_prec = getcontext().prec

    with localcontext() as context:
        context.prec = 2
        rendered = render_monthly_report(report)
        # Still inside the hostile context: no mutation happened.
        assert context.prec == 2
        assert getcontext().prec == 2

    assert getcontext().prec == default_prec
    assert f"Доход: {digits} USDT" in rendered
    assert f"Итог: +{digits} USDT" in rendered
    assert "E+" not in rendered


def test_hostile_context_huge_plus_tiny() -> None:
    """prec=2 ambient context cannot round a huge-then-tiny report value."""
    huge = Decimal(999999999999999999999999999999999999)
    report = build_report(
        year=2026,
        month=8,
        income=huge,
        expense=Decimal("0.000001"),
        net=Decimal("999999999999999999999999999999998.999999"),
        count=2,
    )
    with localcontext() as context:
        context.prec = 2
        rendered = render_monthly_report(report)

    assert "Доход: 999999999999999999999999999999999999 USDT" in rendered
    assert "Расход: 0.000001 USDT" in rendered


# ---------------------------------------------------------------------------
# label safety
# ---------------------------------------------------------------------------


def test_label_escaping_control_characters() -> None:
    """Dangerous label controls become visible escapes; text is preserved."""
    category_label = "Кат\\его\tрия"
    source_label = "Ис\nточник\r"
    report = build_report(
        year=2026,
        month=8,
        income=1,
        expense=0,
        net=1,
        count=1,
        categories=(
            build_category(
                category_label,
                income=1,
                expense=0,
                net=1,
                count=1,
                sources=(build_source(source_label, income=1, net=1, count=1),),
            ),
        ),
    )

    rendered = render_monthly_report(report)
    lines = render_lines(rendered)

    # The category label occupies exactly one logical line with a visible escape
    # (one input backslash renders as a double backslash).
    assert "📂 Кат\\\\его\\tрия" in lines
    # The source label likewise: no real newline or carriage return leaks.
    assert "• Ис\\nточник\\r" in lines
    assert "  доход 1 · расход 0 · итог +1 USDT · операций 1" in lines
    # No raw control characters in the output.
    for line in lines:
        assert "\t" not in line
        assert "\r" not in line
        assert "\n" not in line


def test_label_escaping_c0_and_del() -> None:
    """NUL, other C0 controls, and DEL render as \\uXXXX uppercase escapes."""
    report = build_report(
        year=2026,
        month=8,
        income=1,
        expense=0,
        net=1,
        count=1,
        categories=(
            build_category(
                "Кот\0egory",
                income=1,
                expense=0,
                net=1,
                count=1,
                sources=(build_source("Со\u0001ur\u007Fce", income=1, net=1, count=1),),
            ),
        ),
    )

    rendered = render_monthly_report(report)

    assert "📂 Кот\\u0000egory" in render_lines(rendered)
    assert "• Со\\u0001ur\\u007Fce" in render_lines(rendered)
    assert "  доход 1 · расход 0 · итог +1 USDT · операций 1" in render_lines(rendered)
    # No raw C0 control or DEL character appears anywhere in the output.
    for line in render_lines(rendered):
        for character in line:
            assert ord(character) >= 0x20


def test_labels_preserved_verbatim() -> None:
    """Normal Cyrillic, Latin, spaces, and punctuation stay unchanged."""
    report = build_report(
        year=2026,
        month=8,
        income=1,
        expense=0,
        net=1,
        count=2,
        categories=(
            build_category(
                "Работа (основная)",
                income=1,
                expense=0,
                net=1,
                count=2,
                sources=(
                    build_source("Проект A, v2", income=1, net=1, count=2),
                ),
            ),
        ),
    )

    rendered = render_monthly_report(report)

    assert "📂 Работа (основная)" in render_lines(rendered)
    assert "• Проект A, v2" in render_lines(rendered)


def test_label_escaping_exact_policy() -> None:
    """The exact escape policy: backslash first, then named, then \\uXXXX."""
    label = "a\\b\rc\nd\te\u0000f"
    report = build_report(
        year=2026,
        month=8,
        income=1,
        expense=0,
        net=1,
        count=1,
        categories=(
            build_category(
                label,
                income=1,
                expense=0,
                net=1,
                count=1,
                sources=(build_source("x", income=1, net=1, count=1),),
            ),
        ),
    )

    rendered = render_monthly_report(report)

    assert "📂 a\\\\b\\rc\\nd\\te\\u0000f" in render_lines(rendered)


# ---------------------------------------------------------------------------
# immutability and side effects
# ---------------------------------------------------------------------------


def test_render_does_not_mutate_report() -> None:
    """The report object, its categories, and its sources stay unchanged."""
    categories = (
        build_category(
            "Инфраструктура",
            income=0,
            expense=16,
            net=-16,
            count=1,
            sources=(build_source("Сервер A", expense=16, net=-16, count=1),),
        ),
        build_category(
            "Работа",
            income=5,
            expense=0,
            net=5,
            count=1,
            sources=(build_source("Проект A", income=5, net=5, count=1),),
        ),
    )
    report = build_report(
        year=2026,
        month=8,
        income=5,
        expense=16,
        net=-11,
        count=2,
        categories=categories,
    )
    expected = build_report(
        year=2026,
        month=8,
        income=5,
        expense=16,
        net=-11,
        count=2,
        categories=categories,
    )
    categories_before = report.categories
    sources_before = report.categories[0].sources

    rendered = render_monthly_report(report)

    assert report == expected
    assert report.categories is categories_before
    assert report.categories == categories
    assert report.categories[0].sources is sources_before
    assert report.categories[0].sources == sources_before
    assert report.categories[1].sources == (
        build_source("Проект A", income=5, net=5, count=1),
    )
    # Rendering twice yields the identical string (pure function).
    assert render_monthly_report(report) == rendered


# ---------------------------------------------------------------------------
# ordering: no renderer-side sorting
# ---------------------------------------------------------------------------


def test_no_reorder_of_categories_and_sources() -> None:
    """A deliberately non-lexical order is preserved exactly."""
    report = build_report(
        year=2026,
        month=8,
        income=3,
        expense=0,
        net=3,
        count=3,
        categories=(
            build_category(
                "zeta",
                income=1,
                expense=0,
                net=1,
                count=1,
                sources=(
                    build_source("yankee", income=1, net=1, count=1),
                    build_source("Alpha", income=0, net=0, count=0),
                ),
            ),
            build_category(
                "Alpha",
                income=2,
                expense=0,
                net=2,
                count=2,
                sources=(
                    build_source("mike", income=1, net=1, count=1),
                    build_source("bravo", income=1, net=1, count=1),
                ),
            ),
        ),
    )

    rendered = render_monthly_report(report)
    lines = render_lines(rendered)

    # Category order is the supplied tuple order, not lexical.
    assert lines.index("📂 zeta") < lines.index("📂 Alpha")
    # Source order is the supplied tuple order inside each category.
    zeta_block = lines[lines.index("📂 zeta") : lines.index("📂 Alpha")]
    assert zeta_block.index("• yankee") < zeta_block.index("• Alpha")
    assert zeta_block.index(
        "  доход 1 · расход 0 · итог +1 USDT · операций 1"
    ) < zeta_block.index(
        "  доход 0 · расход 0 · итог 0 USDT · операций 0"
    )
    alpha_block = lines[lines.index("📂 Alpha") :]
    assert alpha_block.index("• mike") < alpha_block.index("• bravo")


# ---------------------------------------------------------------------------
# no recalculation: presentation only
# ---------------------------------------------------------------------------


def test_no_recalculation_of_inconsistent_report() -> None:
    """A manually inconsistent report renders its stored values verbatim.

    Monthly income field is 999 while the child category income is 1:
    the renderer displays 999 at month level and 1 at category level,
    never re-summing or "fixing" the report.
    """
    report = build_report(
        year=2026,
        month=8,
        income=999,
        expense=7,
        net=992,
        count=3,
        categories=(
            build_category(
                "Работа",
                income=1,
                expense=7,
                net=-6,
                count=3,
                sources=(
                    build_source("Проект A", income=1, expense=7, net=-6, count=3),
                ),
            ),
        ),
    )

    rendered = render_monthly_report(report)

    lines = render_lines(rendered)
    assert "📈 Доход: 999 USDT" in lines
    assert "⚖️ Итог: +992 USDT" in lines
    assert "🧾 Операций: 3" in lines
    # The category block shows the stored (inconsistent) values.
    category_index = lines.index("📂 Работа")
    assert lines[category_index + 1] == "Доход: 1 USDT"
    assert lines[category_index + 2] == "Расход: 7 USDT"
    assert lines[category_index + 3] == "Итог: -6 USDT"
    assert lines[category_index + 4] == "Операций: 3"
    assert "• Проект A" in lines
    assert "  доход 1 · расход 7 · итог -6 USDT · операций 3" in lines
    # The renderer never summed the child report to 999 elsewhere.
    assert "Доход: 8 USDT" not in lines


# ---------------------------------------------------------------------------
# input contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_report",
    [None, "report", 42, object(), {"year": 2026}],
)
def test_non_report_rejected_with_type_error(bad_report: Any) -> None:
    """Only a real MonthlyFinanceReport is accepted."""
    with pytest.raises(TypeError):
        render_monthly_report(bad_report)


def test_duck_typed_lookalike_rejected() -> None:
    """An object with the same attributes is still rejected."""

    class Lookalike:
        year = 2026
        month = 8
        income_usdt = Decimal(60)
        expense_usdt = Decimal(21)
        net_usdt = Decimal(39)
        transaction_count = 6
        categories = ()

    with pytest.raises(TypeError):
        render_monthly_report(Lookalike())  # type: ignore[arg-type]


def test_non_finite_decimal_rejected() -> None:
    """Non-finite money values cannot come from E1 and are rejected."""
    report = build_report(
        year=2026, month=8, income=Decimal("NaN"), expense=0, net=0, count=0
    )
    with pytest.raises(TypeError):
        render_monthly_report(report)


# ---------------------------------------------------------------------------
# E1 current-state integration: edit and soft delete
# ---------------------------------------------------------------------------


def test_e2e_edit_current_state_flows_through(conn: sqlite3.Connection) -> None:
    """An edited transaction renders with its edited values."""
    persisted = persist_row(
        conn,
        transaction_date=date(2026, 8, 15),
        amount_usdt=Decimal(25),
        category="Работа",
        source="Проект A",
    )
    edit_transaction(
        conn,
        persisted.transaction_id or "",
        direction=Direction.EXPENSE,
        amount_usdt=Decimal(40),
        category="Инфраструктура",
        source="Сервер B",
        comment=None,
        transaction_date=date(2026, 8, 20),
        updated_at=FIXED_UPDATED_AT,
    )

    rendered = render_monthly_report(build_monthly_report(conn, year=2026, month=8))

    assert rendered == (
        "💰 FINANCE | АВГУСТ 2026\n"
        "\n"
        "📈 Доход: 0 USDT\n"
        "📉 Расход: 40 USDT\n"
        "⚖️ Итог: -40 USDT\n"
        "🧾 Операций: 1\n"
        "\n"
        "📂 Инфраструктура\n"
        "Доход: 0 USDT\n"
        "Расход: 40 USDT\n"
        "Итог: -40 USDT\n"
        "Операций: 1\n"
        "\n"
        "🔹 Источники\n"
        "• Сервер B\n"
        "  доход 0 · расход 40 · итог -40 USDT · операций 1"
    )


def test_e2e_soft_delete_shows_empty_month(conn: sqlite3.Connection) -> None:
    """A soft-deleted transaction contributes nothing to the rendered text."""
    persisted = persist_row(
        conn,
        transaction_date=date(2026, 8, 15),
        amount_usdt=Decimal(25),
        category="Работа",
        source="Проект A",
    )
    soft_delete_transaction(conn, persisted.transaction_id or "", deleted_at=FIXED_DELETED_AT)

    rendered = render_monthly_report(build_monthly_report(conn, year=2026, month=8))

    assert rendered == (
        "💰 FINANCE | АВГУСТ 2026\n"
        "\n"
        "📈 Доход: 0 USDT\n"
        "📉 Расход: 0 USDT\n"
        "⚖️ Итог: 0 USDT\n"
        "🧾 Операций: 0\n"
        "\n"
        "📭 Операций за месяц нет."
    )


def test_e2e_soft_delete_removes_contribution(conn: sqlite3.Connection) -> None:
    """Soft-deleting one row removes exactly its contribution from the render."""
    persist_canonical_august(conn)
    extra = persist_row(
        conn,
        transaction_date=date(2026, 8, 25),
        amount_usdt=Decimal(100),
        category="Бонус",
        source="Gift",
    )
    before = render_monthly_report(build_monthly_report(conn, year=2026, month=8))
    assert "Доход: 160 USDT" in before
    assert "Операций: 7" in before
    assert "📂 Бонус" in before

    soft_delete_transaction(conn, extra.transaction_id or "", deleted_at=FIXED_DELETED_AT)
    after = render_monthly_report(build_monthly_report(conn, year=2026, month=8))

    assert after == CANONICAL_AUGUST


# ---------------------------------------------------------------------------
# public exports and module hygiene
# ---------------------------------------------------------------------------


def test_public_exports() -> None:
    assert hermes_finance.render_monthly_report is rendering_module.render_monthly_report
    assert "render_monthly_report" in hermes_finance.__all__
    for name in hermes_finance.__all__:
        assert not name.startswith("_")
    # Private renderer helpers stay private.
    for name in rendering_module.__all__:
        assert not name.startswith("_")
    # G1: the transaction-list renderers join the public surface.
    # G2: the filtered monthly summary renderers join the public surface.
    assert rendering_module.__all__ == [
        "render_monthly_category_summary",
        "render_monthly_report",
        "render_monthly_source_summary",
        "render_recent_transactions",
        "render_transactions_by_date",
        "render_transactions_by_month",
    ]
    assert hermes_finance.render_recent_transactions is render_recent_transactions
    assert hermes_finance.render_transactions_by_date is render_transactions_by_date
    assert hermes_finance.render_transactions_by_month is render_transactions_by_month
    from hermes_finance import (
        render_monthly_category_summary,
        render_monthly_source_summary,
    )

    assert (
        hermes_finance.render_monthly_category_summary
        is render_monthly_category_summary
    )
    assert (
        hermes_finance.render_monthly_source_summary is render_monthly_source_summary
    )


def test_rendering_source_contains_no_database_clock_or_markup() -> None:
    """The renderer module is pure: no SQL, no clock, no transport markup."""
    for token in ("sqlite", "SELECT", "INSERT", "UPDATE", "DELETE", "PRAGMA"):
        assert token not in RENDERING_SOURCE
    for token in ("execute(", "executemany(", "commit(", "rollback(", "cursor(", "connect("):
        assert token not in RENDERING_SOURCE
    for token in (
        "datetime.now",
        "datetime.today",
        "import time",
        "time.time",
        "time.monotonic",
        "utcnow",
        "today(",
    ):
        assert token not in RENDERING_SOURCE
    for token in ("FinanceConfig", "open(", "read_text(", "write_text(", "Path("):
        assert token not in RENDERING_SOURCE
    for token in ("<b>", "<i>", "parse_mode", "MarkdownV2", "requests", "aiohttp"):
        assert token not in RENDERING_SOURCE
    # No arithmetic on money values: presentation only.
    for token in ("normalize(", "quantize(", "float(", "round(", "+ 1", "sum("):
        assert token not in RENDERING_SOURCE


def test_rendering_uses_only_allowed_imports() -> None:
    """Only stdlib machinery plus the domain and E1 report types are imported."""

    def normalize_imports(source: str) -> list[str]:
        """Collapse parenthesized imports into one canonical line each."""
        statements: list[str] = []
        current = ""
        depth = 0
        for line in source.splitlines():
            if depth == 0 and not line.startswith(("import ", "from ")):
                continue
            current = f"{current} {line.strip()}" if current else line.strip()
            depth += line.count("(") - line.count(")")
            if depth == 0:
                flattened = " ".join(current.split())
                flattened = flattened.replace("(", " ").replace(")", "")
                flattened = " ".join(flattened.split()).rstrip(" ,")
                flattened = flattened.replace(", ", ",").replace(",", ", ")
                statements.append(flattened)
                current = ""
        return statements

    assert normalize_imports(RENDERING_SOURCE) == [
        "from __future__ import annotations",
        "from collections.abc import Sequence",
        "from datetime import date",
        "from decimal import Decimal",
        "from typing import Final",
        (
            "from hermes_finance.domain import Direction, Transaction,"
            " normalize_required_text, require_calendar_date"
        ),
        (
            "from hermes_finance.reporting import CategoryReport,"
            " MonthlyFinanceReport, SourceReport"
        ),
    ]


# ---------------------------------------------------------------------------
# G1: deterministic transaction-list rendering
# ---------------------------------------------------------------------------


def make_stored_transaction(**overrides: Any) -> Transaction:
    """A valid persisted Transaction (real id) with field overrides."""
    values: dict[str, Any] = {
        "direction": Direction.INCOME,
        "amount_usdt": Decimal(25),
        "category": "Работа",
        "source": "Проект A",
        "comment": None,
        "transaction_date": date(2026, 9, 5),
        "created_at": FIXED_CREATED_AT,
        "updated_at": FIXED_UPDATED_AT,
        "transaction_id": "15",
        "status": TransactionStatus.ACTIVE,
        "deleted_at": None,
    }
    values.update(overrides)
    return Transaction(**values)


def test_g1_recent_non_empty_exact_card() -> None:
    """The canonical recent view renders the exact BlackCat card."""
    rendered = render_recent_transactions(
        (
            make_stored_transaction(),
            make_stored_transaction(
                direction=Direction.EXPENSE,
                amount_usdt=Decimal(10),
                category="Инфраструктура",
                source="Сервер B",
                transaction_id="14",
                transaction_date=date(2026, 9, 4),
                comment="продление сервера",
            ),
        )
    )

    assert rendered == (
        "🧾 FINANCE | ПОСЛЕДНИЕ ОПЕРАЦИИ\n"
        "\n"
        "🟢 #15 | +25 USDT | 05.09.2026\n"
        "Работа • Проект A\n"
        "\n"
        "🔴 #14 | -10 USDT | 04.09.2026\n"
        "Инфраструктура • Сервер B\n"
        "💬 продление сервера"
    )
    assert not rendered.endswith("\n")
    assert all(line == line.rstrip() for line in render_lines(rendered))


def test_g1_exact_date_non_empty_exact_card() -> None:
    """The canonical exact-date view renders the DD.MM.YYYY title."""
    rendered = render_transactions_by_date(
        (make_stored_transaction(),),
        date(2026, 9, 5),
    )

    assert rendered == (
        "🧾 FINANCE | ОПЕРАЦИИ 05.09.2026\n"
        "\n"
        "🟢 #15 | +25 USDT | 05.09.2026\n"
        "Работа • Проект A"
    )


def test_g1_month_list_non_empty_exact_card() -> None:
    """The canonical month view renders the uppercase Russian month title."""
    rendered = render_transactions_by_month(
        (make_stored_transaction(),),
        year=2026,
        month=9,
    )

    assert rendered == (
        "🧾 FINANCE | ОПЕРАЦИИ СЕНТЯБРЬ 2026\n"
        "\n"
        "🟢 #15 | +25 USDT | 05.09.2026\n"
        "Работа • Проект A"
    )


@pytest.mark.parametrize(
    ("renderer", "title"),
    [
        (lambda: render_recent_transactions(()), "🧾 FINANCE | ПОСЛЕДНИЕ ОПЕРАЦИИ"),
        (
            lambda: render_transactions_by_date((), date(2026, 9, 5)),
            "🧾 FINANCE | ОПЕРАЦИИ 05.09.2026",
        ),
        (
            lambda: render_transactions_by_month((), year=2026, month=9),
            "🧾 FINANCE | ОПЕРАЦИИ СЕНТЯБРЬ 2026",
        ),
    ],
)
def test_g1_each_empty_state_exact(renderer: Any, title: str) -> None:
    """Every view renders its title plus the single empty marker."""
    rendered = renderer()

    assert rendered == f"{title}\n\n📭 Операций нет."
    assert not rendered.endswith("\n")
    assert all(line == line.rstrip() for line in render_lines(rendered))


def test_g1_income_icon_and_sign() -> None:
    rendered = render_recent_transactions((make_stored_transaction(),))
    assert "🟢 #15 | +25 USDT | 05.09.2026" in render_lines(rendered)
    assert "🔴" not in rendered


def test_g1_expense_icon_and_sign() -> None:
    rendered = render_recent_transactions(
        (
            make_stored_transaction(
                direction=Direction.EXPENSE,
                amount_usdt=Decimal("10.5"),
                transaction_id="3",
            ),
        )
    )
    assert "🔴 #3 | -10.5 USDT | 05.09.2026" in render_lines(rendered)
    assert "🟢" not in rendered


def test_g1_comment_present_renders_comment_line() -> None:
    rendered = render_recent_transactions(
        (make_stored_transaction(comment="продление сервера"),)
    )
    lines = render_lines(rendered)
    assert lines[3] == "Работа • Проект A"
    assert lines[4] == "💬 продление сервера"


def test_g1_comment_absent_renders_no_comment_line() -> None:
    rendered = render_recent_transactions((make_stored_transaction(),))
    assert "💬" not in rendered
    assert render_lines(rendered)[-1] == "Работа • Проект A"


def test_g1_multiple_transactions_preserve_supplied_order() -> None:
    """A deliberately non-sorted input sequence is displayed verbatim."""
    rendered = render_recent_transactions(
        (
            make_stored_transaction(transaction_id="7"),
            make_stored_transaction(transaction_id="2"),
            make_stored_transaction(transaction_id="99"),
        )
    )

    lines = render_lines(rendered)
    id_lines = [line for line in lines if line.startswith(("🟢", "🔴"))]
    assert id_lines == [
        "🟢 #7 | +25 USDT | 05.09.2026",
        "🟢 #2 | +25 USDT | 05.09.2026",
        "🟢 #99 | +25 USDT | 05.09.2026",
    ]


def test_g1_exact_arbitrary_decimal_precision() -> None:
    """A high-precision amount renders exactly, no rounding or E-notation."""
    amount = Decimal("999999999999999999.999999999999999999")
    rendered = render_recent_transactions(
        (make_stored_transaction(amount_usdt=amount),)
    )
    assert "🟢 #15 | +999999999999999999.999999999999999999 USDT" in rendered
    assert "E+" not in rendered
    assert "E-" not in rendered


def test_g1_trailing_zero_and_integral_amounts() -> None:
    rendered = render_recent_transactions(
        (
            make_stored_transaction(amount_usdt=Decimal("25.5000"), transaction_id="1"),
            make_stored_transaction(amount_usdt=Decimal("40.0"), transaction_id="2"),
        )
    )
    assert "+25.5 USDT" in rendered
    assert "25.5000" not in rendered
    assert "+40 USDT" in rendered
    assert "+40.0" not in rendered


def test_g1_label_and_comment_control_escaping() -> None:
    """Category, source, and comment controls become visible escapes."""
    rendered = render_recent_transactions(
        (
            make_stored_transaction(
                category="Кат\\его\tрия",
                source="Ис\nточник",
                comment="ко\0мментарий",
            ),
        )
    )

    lines = render_lines(rendered)
    assert lines[3] == "Кат\\\\его\\tрия • Ис\\nточник"
    assert lines[4] == "💬 ко\\u0000мментарий"
    for line in lines:
        for character in line:
            assert ord(character) >= 0x20


def test_g1_missing_transaction_id_fails_deterministically() -> None:
    """An unpersisted transaction never renders a fake identifier."""
    unpersisted = make_stored_transaction(transaction_id=None)

    with pytest.raises(ValueError, match="transaction_id"):
        render_recent_transactions((unpersisted,))
    with pytest.raises(ValueError, match="transaction_id"):
        render_transactions_by_date((unpersisted,), date(2026, 9, 5))
    with pytest.raises(ValueError, match="transaction_id"):
        render_transactions_by_month((unpersisted,), year=2026, month=9)


def test_g1_transaction_id_displayed_as_persisted_string() -> None:
    """The persisted id string is used verbatim, never as a number."""
    rendered = render_recent_transactions(
        (make_stored_transaction(transaction_id="0042"),)
    )
    assert "🟢 #0042 | +25 USDT" in rendered


def test_g1_deterministic_repeat_render() -> None:
    transactions = (
        make_stored_transaction(),
        make_stored_transaction(
            direction=Direction.EXPENSE,
            amount_usdt=Decimal(10),
            category="Инфраструктура",
            source="Сервер B",
            transaction_id="14",
            transaction_date=date(2026, 9, 4),
            comment="продление сервера",
        ),
    )

    first = render_recent_transactions(transactions)
    second = render_recent_transactions(transactions)

    assert first == second


@pytest.mark.parametrize(
    "bad_transactions",
    [None, "transactions", 42, object(), (make_stored_transaction(), "not-a-transaction")],
)
def test_g1_non_transaction_input_rejected(bad_transactions: Any) -> None:
    with pytest.raises(TypeError):
        render_recent_transactions(bad_transactions)


def test_g1_by_date_requires_real_date() -> None:
    with pytest.raises(TypeError):
        render_transactions_by_date(
            (),
            datetime(2026, 9, 5, 0, 0, 0, tzinfo=FIXED_TZ),
        )
    with pytest.raises(TypeError):
        render_transactions_by_date((), "2026-09-05")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_month", [0, 13, "9", None, True])
def test_g1_month_renderer_rejects_invalid_month(bad_month: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        render_transactions_by_month((), year=2026, month=bad_month)


@pytest.mark.parametrize("bad_year", ["2026", None, True])
def test_g1_month_renderer_rejects_invalid_year(bad_year: Any) -> None:
    with pytest.raises(TypeError):
        render_transactions_by_month((), year=bad_year, month=9)


def test_g1_end_to_end_db_recent_through_accepted_d1(
    conn: sqlite3.Connection,
) -> None:
    """DB -> D1 -> renderer: persisted rows render with their real ids."""
    persist_row(conn, transaction_date=date(2026, 9, 5), amount_usdt=Decimal(25))
    persist_row(
        conn,
        transaction_date=date(2026, 9, 4),
        direction=Direction.EXPENSE,
        amount_usdt=Decimal(10),
        category="Инфраструктура",
        source="Сервер B",
        comment="продление сервера",
    )

    rendered = render_recent_transactions(
        list_recent_transactions(conn, limit=10)
    )

    lines = render_lines(rendered)
    # D1 newest-first ordering is inherited verbatim: the 05.09 row first.
    assert lines[0] == "🧾 FINANCE | ПОСЛЕДНИЕ ОПЕРАЦИИ"
    assert lines[2].startswith("🟢 #")
    assert lines[2].endswith(" | +25 USDT | 05.09.2026")
    assert lines[5].startswith("🔴 #")
    assert lines[5].endswith(" | -10 USDT | 04.09.2026")
    assert lines[7] == "💬 продление сервера"


def test_g1_end_to_end_db_date_and_month_through_accepted_d1(
    conn: sqlite3.Connection,
) -> None:
    persist_row(conn, transaction_date=date(2026, 9, 5))
    persist_row(conn, transaction_date=date(2026, 8, 20))

    by_date = render_transactions_by_date(
        list_transactions_by_date(conn, date(2026, 9, 5)),
        date(2026, 9, 5),
    )
    assert render_lines(by_date)[0] == "🧾 FINANCE | ОПЕРАЦИИ 05.09.2026"
    assert "05.09.2026" in by_date
    assert "20.08.2026" not in by_date

    by_month = render_transactions_by_month(
        list_transactions_by_month(conn, year=2026, month=9),
        year=2026,
        month=9,
    )
    assert render_lines(by_month)[0] == "🧾 FINANCE | ОПЕРАЦИИ СЕНТЯБРЬ 2026"
    assert "05.09.2026" in by_month
    assert "20.08.2026" not in by_month


def test_g1_soft_deleted_rows_are_invisible_through_d1(
    conn: sqlite3.Connection,
) -> None:
    """D1 ACTIVE-only visibility flows through the renderer unchanged."""
    persisted = persist_row(conn, transaction_date=date(2026, 9, 5))
    soft_delete_transaction(
        conn, persisted.transaction_id or "", deleted_at=FIXED_DELETED_AT
    )

    rendered = render_recent_transactions(list_recent_transactions(conn, limit=10))

    assert rendered == (
        "🧾 FINANCE | ПОСЛЕДНИЕ ОПЕРАЦИИ\n"
        "\n"
        "📭 Операций нет."
    )
