"""Tests for the filtered monthly summary layer (stage G2).

All tests are deterministic: no network, no wall-clock time, no
randomness, no environment access, no subprocesses. The selection layer
is exercised as a pure projection over directly constructed immutable
E1 report objects; the integration facades are exercised end-to-end
through the real accepted parser-free persistence path (real in-memory
SQLite, real E1 aggregation, real G2 selection, real renderer). No
lower-layer business logic is re-implemented or stubbed here.
"""

from __future__ import annotations

import inspect
import sqlite3
from collections.abc import Iterator
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

import pytest

import hermes_finance
import hermes_finance.filtered_summary as filtered_summary_module
import hermes_finance.integration as integration_module
from hermes_finance import (
    CategoryReport,
    Direction,
    FinanceConfig,
    MonthlyFinanceReport,
    SourceReport,
    Transaction,
    TransactionStatus,
    build_monthly_report,
    get_monthly_category_summary_text,
    get_monthly_source_summary_text,
    open_database,
    persist_transaction,
    render_monthly_category_summary,
    render_monthly_source_summary,
    select_category_source,
    select_monthly_category,
    soft_delete_transaction,
)
from hermes_finance.provenance import TelegramMessageRef
from hermes_finance.repository import RepositoryDataError

FILTERED_SOURCE: Final[str] = Path(
    filtered_summary_module.__file__
).read_text(encoding="utf-8")

FIXED_TZ = UTC
FIXED_CREATED_AT = datetime(2026, 9, 1, 10, 0, 0, tzinfo=FIXED_TZ)
FIXED_PROCESSED_AT = datetime(2026, 9, 1, 12, 15, 0, tzinfo=FIXED_TZ)

ZERO: Final[Decimal] = Decimal(0)

#: The category card of the canonical G2 example (spec: Работа,
#: September 2026: income 125, expense 20, net +105, 7 operations).
CATEGORY_CARD: Final[str] = (
    "💰 FINANCE | СЕНТЯБРЬ 2026\n"
    "\n"
    "🔎 Категория: Работа\n"
    "\n"
    "📈 Доход: 125 USDT\n"
    "📉 Расход: 20 USDT\n"
    "⚖️ Итог: +105 USDT\n"
    "🧾 Операций: 7"
)

#: The category+source card of the canonical G2 example (spec:
#: Работа / Проект A: income 80, expense 10, net +70, 4 operations).
SOURCE_CARD: Final[str] = (
    "💰 FINANCE | СЕНТЯБРЬ 2026\n"
    "\n"
    "🔎 Работа • Проект A\n"
    "\n"
    "📈 Доход: 80 USDT\n"
    "📉 Расход: 10 USDT\n"
    "⚖️ Итог: +70 USDT\n"
    "🧾 Операций: 4"
)


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


def make_ref(*, message_id: int) -> TelegramMessageRef:
    """A valid Telegram message provenance reference."""
    return TelegramMessageRef(
        chat_id=-100200,
        message_thread_id=7,
        message_id=message_id,
        update_id=10_000 + message_id,
    )


def make_transaction(**overrides: Any) -> Transaction:
    """A valid unpersisted Transaction with field overrides."""
    values: dict[str, Any] = {
        "direction": Direction.INCOME,
        "amount_usdt": Decimal(25),
        "category": "Работа",
        "source": "Проект A",
        "comment": None,
        "transaction_date": date(2026, 9, 5),
        "created_at": FIXED_CREATED_AT,
        "updated_at": FIXED_CREATED_AT,
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
        make_ref(message_id=message_id),
        processed_at=FIXED_PROCESSED_AT,
    )


def make_report() -> MonthlyFinanceReport:
    """The canonical G2 report: Работа and Сервисы, September 2026.

    The source label ``Проект A`` deliberately exists under BOTH
    categories with different values, pinning that G2 selection stays
    category-local and never merges sources globally.
    """
    return MonthlyFinanceReport(
        year=2026,
        month=9,
        income_usdt=Decimal(139),
        expense_usdt=Decimal(23),
        net_usdt=Decimal(116),
        transaction_count=9,
        categories=(
            CategoryReport(
                category="Сервисы",
                income_usdt=Decimal(14),
                expense_usdt=Decimal(3),
                net_usdt=Decimal(11),
                transaction_count=2,
                sources=(
                    SourceReport(
                        source="Проект A",
                        income_usdt=Decimal(14),
                        expense_usdt=Decimal(3),
                        net_usdt=Decimal(11),
                        transaction_count=2,
                    ),
                ),
            ),
            CategoryReport(
                category="Работа",
                income_usdt=Decimal(125),
                expense_usdt=Decimal(20),
                net_usdt=Decimal(105),
                transaction_count=7,
                sources=(
                    SourceReport(
                        source="Проект B",
                        income_usdt=Decimal(45),
                        expense_usdt=Decimal(10),
                        net_usdt=Decimal(35),
                        transaction_count=3,
                    ),
                    SourceReport(
                        source="Проект A",
                        income_usdt=Decimal(80),
                        expense_usdt=Decimal(10),
                        net_usdt=Decimal(70),
                        transaction_count=4,
                    ),
                ),
            ),
        ),
    )


def work_report(report: MonthlyFinanceReport) -> CategoryReport:
    """The Работа category report of ``report``."""
    selected = select_monthly_category(report, category="Работа")
    assert selected is not None
    return selected


def persist_canonical_september(connection: sqlite3.Connection) -> None:
    """Persist the canonical seven-transaction Работа month (2026-09).

    Работа / Проект A: +50, +30, -6, -4 (income 80, expense 10, 4 ops).
    Работа / Проект B: +25, +20, -10 (income 45, expense 10, 3 ops).
    """
    rows: list[tuple[str, str, Decimal]] = [
        ("Проект A", "income", Decimal(50)),
        ("Проект A", "income", Decimal(30)),
        ("Проект A", "expense", Decimal(6)),
        ("Проект A", "expense", Decimal(4)),
        ("Проект B", "income", Decimal(25)),
        ("Проект B", "income", Decimal(20)),
        ("Проект B", "expense", Decimal(10)),
    ]
    for source, direction, amount in rows:
        persist_row(
            connection,
            transaction_date=date(2026, 9, 5),
            direction=Direction.INCOME if direction == "income" else Direction.EXPENSE,
            amount_usdt=amount,
            category="Работа",
            source=source,
        )


def render_lines(rendered: str) -> list[str]:
    """Split a rendered view into lines for per-line assertions."""
    return rendered.split("\n")


# ---------------------------------------------------------------------------
# SELECTION: exact category matching
# ---------------------------------------------------------------------------


def test_category_exact_match_returns_the_authoritative_report() -> None:
    """An exact label selects the very report object held by E1."""
    report = make_report()

    selected = select_monthly_category(report, category="Работа")

    assert selected is report.categories[1]
    assert selected.category == "Работа"


def test_category_missing_returns_none() -> None:
    """A label absent from the month is a valid no-result, not an error."""
    report = make_report()

    assert select_monthly_category(report, category="Инфраструктура") is None


def test_category_missing_in_empty_month_returns_none() -> None:
    """A valid empty month has no categories and selects nothing."""
    report = MonthlyFinanceReport(
        year=2026,
        month=9,
        income_usdt=ZERO,
        expense_usdt=ZERO,
        net_usdt=ZERO,
        transaction_count=0,
        categories=(),
    )

    assert select_monthly_category(report, category="Работа") is None


@pytest.mark.parametrize(
    "label", ["работа", "РАБОТА", "РаБота", "Работа.", "Рабо та"]
)
def test_category_matching_stays_case_sensitive_and_exact(label: str) -> None:
    """No casefolding, lowercasing, fuzzy matching, or aliasing."""
    report = make_report()

    assert select_monthly_category(report, category=label) is None


@pytest.mark.parametrize("label", [" Работа", "Работа ", "\tРабота\n", "  Работа  "])
def test_category_surrounding_whitespace_is_normalized(label: str) -> None:
    """The accepted required-text strip is the only normalization."""
    report = make_report()

    selected = select_monthly_category(report, category=label)

    assert selected is not None
    assert selected.category == "Работа"


@pytest.mark.parametrize("bad_category", [None, 5, b"bytes-label", object()])
def test_category_non_string_rejected(bad_category: Any) -> None:
    report = make_report()

    with pytest.raises(TypeError):
        select_monthly_category(report, category=bad_category)


@pytest.mark.parametrize("blank_category", ["", " ", "\t", "\n", "  \t "])
def test_category_blank_rejected(blank_category: str) -> None:
    report = make_report()

    with pytest.raises(ValueError):
        select_monthly_category(report, category=blank_category)


@pytest.mark.parametrize(
    "bad_report", [None, "report", 42, object(), ("Работа",)]
)
def test_category_non_report_rejected(bad_report: Any) -> None:
    with pytest.raises(TypeError):
        select_monthly_category(bad_report, category="Работа")


# ---------------------------------------------------------------------------
# SELECTION: exact category-local source matching
# ---------------------------------------------------------------------------


def test_source_exact_match_inside_category() -> None:
    """An exact source label selects the very report object held by E1."""
    report = make_report()
    work = work_report(report)

    selected = select_category_source(work, source="Проект A")

    assert selected is work.sources[1]
    assert selected.source == "Проект A"


def test_source_missing_inside_existing_category_returns_none() -> None:
    report = make_report()
    work = work_report(report)

    assert select_category_source(work, source="Сервер B") is None


@pytest.mark.parametrize(    "label", ["проект a", "ПРОЕКТ A", "Проект А", "Проект A!"])
def test_source_matching_stays_case_sensitive_and_exact(label: str) -> None:
    report = make_report()
    work = work_report(report)

    assert select_category_source(work, source=label) is None


@pytest.mark.parametrize("label", [" Проект A", "Проект A ", "\tПроект A\n"])
def test_source_surrounding_whitespace_is_normalized(label: str) -> None:
    report = make_report()
    work = work_report(report)

    selected = select_category_source(work, source=label)

    assert selected is not None
    assert selected.source == "Проект A"


@pytest.mark.parametrize("bad_source", [None, 5, b"bytes-label", object()])
def test_source_non_string_rejected(bad_source: Any) -> None:
    report = make_report()
    work = work_report(report)

    with pytest.raises(TypeError):
        select_category_source(work, source=bad_source)


@pytest.mark.parametrize("blank_source", ["", " ", "\t", "\n"])
def test_source_blank_rejected(blank_source: str) -> None:
    report = make_report()
    work = work_report(report)

    with pytest.raises(ValueError):
        select_category_source(work, source=blank_source)


@pytest.mark.parametrize(
    "bad_category_report", [None, "category", 42, object(), make_report()]
)
def test_source_non_category_report_rejected(bad_category_report: Any) -> None:
    with pytest.raises(TypeError):
        select_category_source(bad_category_report, source="Проект A")


def test_same_source_under_two_categories_is_not_globally_merged() -> None:
    """«Проект A» exists under both categories with different values."""
    report = make_report()

    work_project_a = select_category_source(
        work_report(report), source="Проект A"
    )
    services_project_a = select_category_source(
        report.categories[0], source="Проект A"
    )

    assert work_project_a is not None
    assert services_project_a is not None
    assert work_project_a is not services_project_a
    assert work_project_a.income_usdt == Decimal(80)
    assert services_project_a.income_usdt == Decimal(14)
    assert work_project_a.transaction_count == 4
    assert services_project_a.transaction_count == 2


def test_selection_never_recalculates_and_preserves_e1_values_verbatim() -> None:
    """The selected objects ARE the E1 objects: identity, not copies."""
    report = make_report()

    work = select_monthly_category(report, category="Работа")
    assert work is not None
    assert work.income_usdt is report.categories[1].income_usdt
    assert work.expense_usdt is report.categories[1].expense_usdt
    assert work.net_usdt is report.categories[1].net_usdt
    assert work.transaction_count == report.categories[1].transaction_count

    project_a = select_category_source(work, source="Проект A")
    assert project_a is not None
    assert project_a is report.categories[1].sources[1]
    assert project_a.income_usdt is report.categories[1].sources[1].income_usdt
    assert project_a.net_usdt is report.categories[1].sources[1].net_usdt


# ---------------------------------------------------------------------------
# RENDERING: exact BlackCat cards
# ---------------------------------------------------------------------------


def test_category_card_exact_string() -> None:
    report = make_report()
    work = work_report(report)

    rendered = render_monthly_category_summary(
        year=2026,
        month=9,
        category="Работа",
        category_report=work,
    )

    assert rendered == CATEGORY_CARD
    assert not rendered.endswith("\n")
    assert all(line == line.rstrip() for line in render_lines(rendered))


def test_source_card_exact_string() -> None:
    report = make_report()
    project_a = select_category_source(work_report(report), source="Проект A")
    assert project_a is not None

    rendered = render_monthly_source_summary(
        year=2026,
        month=9,
        category="Работа",
        source="Проект A",
        source_report=project_a,
    )

    assert rendered == SOURCE_CARD
    assert not rendered.endswith("\n")
    assert all(line == line.rstrip() for line in render_lines(rendered))


def test_category_not_found_card_exact_string() -> None:
    rendered = render_monthly_category_summary(
        year=2026,
        month=9,
        category="Инфраструктура",
        category_report=None,
    )

    assert rendered == (
        "💰 FINANCE | СЕНТЯБРЬ 2026\n"
        "\n"
        "🔎 Категория: Инфраструктура\n"
        "\n"
        "📭 Данных нет."
    )


def test_source_not_found_inside_existing_category_card() -> None:
    rendered = render_monthly_source_summary(
        year=2026,
        month=9,
        category="Работа",
        source="Сервер B",
        source_report=None,
    )

    assert rendered == (
        "💰 FINANCE | СЕНТЯБРЬ 2026\n"
        "\n"
        "🔎 Работа • Сервер B\n"
        "\n"
        "📭 Данных нет."
    )


def test_source_not_found_because_category_missing_card() -> None:
    """A missing category renders the same explicit no-data state."""
    rendered = render_monthly_source_summary(
        year=2026,
        month=9,
        category="Инфраструктура",
        source="Проект A",
        source_report=None,
    )

    assert rendered == (
        "💰 FINANCE | СЕНТЯБРЬ 2026\n"
        "\n"
        "🔎 Инфраструктура • Проект A\n"
        "\n"
        "📭 Данных нет."
    )


def test_existing_zero_totals_render_zeros_not_no_data() -> None:
    """An existing label with exact zero totals is NOT the no-data state."""
    empty_category = CategoryReport(
        category="Пусто",
        income_usdt=ZERO,
        expense_usdt=ZERO,
        net_usdt=ZERO,
        transaction_count=0,
        sources=(),
    )

    rendered = render_monthly_category_summary(
        year=2026,
        month=9,
        category="Пусто",
        category_report=empty_category,
    )

    assert rendered == (
        "💰 FINANCE | СЕНТЯБРЬ 2026\n"
        "\n"
        "🔎 Категория: Пусто\n"
        "\n"
        "📈 Доход: 0 USDT\n"
        "📉 Расход: 0 USDT\n"
        "⚖️ Итог: 0 USDT\n"
        "🧾 Операций: 0"
    )


def test_positive_negative_and_zero_net_signs() -> None:
    def category_with(net: Decimal) -> CategoryReport:
        return CategoryReport(
            category="Категория",
            income_usdt=Decimal(25),
            expense_usdt=Decimal(10),
            net_usdt=net,
            transaction_count=1,
            sources=(),
        )

    positive = render_monthly_category_summary(
        year=2026, month=9, category="Категория",
        category_report=category_with(Decimal("105.5")),
    )
    assert "⚖️ Итог: +105.5 USDT" in positive

    negative = render_monthly_category_summary(
        year=2026, month=9, category="Категория",
        category_report=category_with(Decimal(-11)),
    )
    assert "⚖️ Итог: -11 USDT" in negative

    zero = render_monthly_category_summary(
        year=2026, month=9, category="Категория",
        category_report=category_with(ZERO),
    )
    assert "⚖️ Итог: 0 USDT" in zero


def test_arbitrary_decimal_precision_rendered_exactly() -> None:
    amount = Decimal("999999999999999999.999999999999999999")
    source = SourceReport(
        source="Проект B",
        income_usdt=amount,
        expense_usdt=Decimal("0.000000000000000001"),
        net_usdt=Decimal("999999999999999999.999999999999999998"),
        transaction_count=1,
    )

    rendered = render_monthly_source_summary(
        year=2026,
        month=9,
        category="Работа",
        source="Проект B",
        source_report=source,
    )

    assert "📈 Доход: 999999999999999999.999999999999999999 USDT" in rendered
    assert "📉 Расход: 0.000000000000000001 USDT" in rendered
    assert "⚖️ Итог: +999999999999999999.999999999999999998 USDT" in rendered
    assert "E+" not in rendered
    assert "E-" not in rendered


def test_trailing_fractional_zeros_trimmed_and_integral_without_point() -> None:
    category = CategoryReport(
        category="Работа",
        income_usdt=Decimal("125.5000"),
        expense_usdt=Decimal("20.0"),
        net_usdt=Decimal("105.500"),
        transaction_count=7,
        sources=(),
    )

    rendered = render_monthly_category_summary(
        year=2026, month=9, category="Работа", category_report=category
    )

    assert "📈 Доход: 125.5 USDT" in rendered
    assert "125.5000" not in rendered
    assert "📉 Расход: 20 USDT" in rendered
    assert "20.0" not in rendered
    assert "⚖️ Итог: +105.5 USDT" in rendered


def test_label_control_characters_are_escaped() -> None:
    """Control characters in labels never break the card layout."""
    rendered = render_monthly_category_summary(
        year=2026,
        month=9,
        category="Кат\\его\tрия",
        category_report=None,
    )

    lines = render_lines(rendered)
    assert lines[2] == "🔎 Категория: Кат\\\\его\\tрия"
    for line in lines:
        for character in line:
            assert ord(character) >= 0x20

    source_rendered = render_monthly_source_summary(
        year=2026,
        month=9,
        category="Кат\\его\tрия",
        source="Ис\nточник",
        source_report=None,
    )
    source_lines = render_lines(source_rendered)
    assert source_lines[2] == "🔎 Кат\\\\его\\tрия • Ис\\nточник"
    for line in source_lines:
        for character in line:
            assert ord(character) >= 0x20


def test_requested_label_surrounding_whitespace_displayed_stripped() -> None:
    rendered = render_monthly_category_summary(
        year=2026,
        month=9,
        category="  Работа  ",
        category_report=None,
    )

    assert render_lines(rendered)[2] == "🔎 Категория: Работа"


def test_deterministic_repeat_rendering() -> None:
    report = make_report()
    work = work_report(report)
    project_a = select_category_source(work, source="Проект A")
    assert project_a is not None

    first_category = render_monthly_category_summary(
        year=2026, month=9, category="Работа", category_report=work
    )
    second_category = render_monthly_category_summary(
        year=2026, month=9, category="Работа", category_report=work
    )
    assert first_category == second_category == CATEGORY_CARD

    first_source = render_monthly_source_summary(
        year=2026, month=9, category="Работа", source="Проект A",
        source_report=project_a,
    )
    second_source = render_monthly_source_summary(
        year=2026, month=9, category="Работа", source="Проект A",
        source_report=project_a,
    )
    assert first_source == second_source == SOURCE_CARD


def test_month_names_are_hard_coded_uppercase_russian() -> None:
    rendered = render_monthly_category_summary(
        year=2026, month=8, category="Работа", category_report=None
    )
    assert render_lines(rendered)[0] == "💰 FINANCE | АВГУСТ 2026"


@pytest.mark.parametrize("bad_month", [0, 13, "9", None, True])
def test_summary_renderer_rejects_invalid_month(bad_month: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        render_monthly_category_summary(
            year=2026, month=bad_month, category="Работа", category_report=None
        )
    with pytest.raises((TypeError, ValueError)):
        render_monthly_source_summary(
            year=2026, month=bad_month, category="Работа",
            source="Проект A", source_report=None,
        )


@pytest.mark.parametrize("bad_year", ["2026", None, True, 2026.0])
def test_summary_renderer_rejects_invalid_year(bad_year: Any) -> None:
    with pytest.raises(TypeError):
        render_monthly_category_summary(
            year=bad_year, month=9, category="Работа", category_report=None
        )
    with pytest.raises(TypeError):
        render_monthly_source_summary(
            year=bad_year, month=9, category="Работа",
            source="Проект A", source_report=None,
        )


@pytest.mark.parametrize("blank_label", ["", " ", "\t"])
def test_summary_renderer_rejects_blank_labels(blank_label: str) -> None:
    with pytest.raises(ValueError):
        render_monthly_category_summary(
            year=2026, month=9, category=blank_label, category_report=None
        )
    with pytest.raises(ValueError):
        render_monthly_source_summary(
            year=2026, month=9, category="Работа",
            source=blank_label, source_report=None,
        )
    with pytest.raises(ValueError):
        render_monthly_source_summary(
            year=2026, month=9, category=blank_label,
            source="Проект A", source_report=None,
        )


@pytest.mark.parametrize("bad_label", [None, 5, b"bytes-label", object()])
def test_summary_renderer_rejects_non_string_labels(bad_label: Any) -> None:
    with pytest.raises(TypeError):
        render_monthly_category_summary(
            year=2026, month=9, category=bad_label, category_report=None
        )
    with pytest.raises(TypeError):
        render_monthly_source_summary(
            year=2026, month=9, category=bad_label, source="Проект A", source_report=None
        )
    with pytest.raises(TypeError):
        render_monthly_source_summary(
            year=2026, month=9, category="Работа",
            source=bad_label, source_report=None,
        )


def test_summary_renderer_rejects_lookalike_report_objects() -> None:
    class FakeCategoryReport:
        category = "Работа"
        income_usdt = ZERO
        expense_usdt = ZERO
        net_usdt = ZERO
        transaction_count = 0
        sources: tuple[SourceReport, ...] = ()

    class FakeSourceReport:
        source = "Проект A"
        income_usdt = ZERO
        expense_usdt = ZERO
        net_usdt = ZERO
        transaction_count = 0

    with pytest.raises(TypeError):
        render_monthly_category_summary(
            year=2026,
            month=9,
            category="Работа",
            category_report=FakeCategoryReport(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        render_monthly_source_summary(
            year=2026,
            month=9,
            category="Работа",
            source="Проект A",
            source_report=FakeSourceReport(),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# INTEGRATION: facade composition over E1 + G2 selection + renderer
# ---------------------------------------------------------------------------


def test_category_facade_matches_manual_composition(conn: sqlite3.Connection) -> None:
    """The facade output equals E1 -> selection -> renderer composed by hand."""
    persist_canonical_september(conn)

    facade_text = get_monthly_category_summary_text(
        conn, year=2026, month=9, category="Работа"
    )
    composed_text = render_monthly_category_summary(
        year=2026,
        month=9,
        category="Работа",
        category_report=select_monthly_category(
            build_monthly_report(conn, year=2026, month=9), category="Работа"
        ),
    )

    assert facade_text == composed_text == CATEGORY_CARD


def test_source_facade_matches_manual_composition(conn: sqlite3.Connection) -> None:
    persist_canonical_september(conn)

    facade_text = get_monthly_source_summary_text(
        conn, year=2026, month=9, category="Работа", source="Проект A"
    )
    report = build_monthly_report(conn, year=2026, month=9)
    work = select_monthly_category(report, category="Работа")
    assert work is not None
    composed_text = render_monthly_source_summary(
        year=2026,
        month=9,
        category="Работа",
        source="Проект A",
        source_report=select_category_source(work, source="Проект A"),
    )

    assert facade_text == composed_text == SOURCE_CARD


def test_category_facade_exact_card_through_real_e1(
    conn: sqlite3.Connection,
) -> None:
    """DB -> E1 -> G2 -> renderer with exact full-string equality."""
    persist_canonical_september(conn)

    assert (
        get_monthly_category_summary_text(
            conn, year=2026, month=9, category="Работа"
        )
        == CATEGORY_CARD
    )
    assert (
        get_monthly_source_summary_text(
            conn, year=2026, month=9, category="Работа", source="Проект A"
        )
        == SOURCE_CARD
    )


def test_category_facade_missing_category_renders_no_data(
    conn: sqlite3.Connection,
) -> None:
    persist_canonical_september(conn)

    assert get_monthly_category_summary_text(
        conn, year=2026, month=9, category="Инфраструктура"
    ) == (
        "💰 FINANCE | СЕНТЯБРЬ 2026\n"
        "\n"
        "🔎 Категория: Инфраструктура\n"
        "\n"
        "📭 Данных нет."
    )


def test_source_facade_missing_source_renders_no_data(
    conn: sqlite3.Connection,
) -> None:
    persist_canonical_september(conn)

    assert get_monthly_source_summary_text(
        conn, year=2026, month=9, category="Работа", source="Сервер B"
    ) == (
        "💰 FINANCE | СЕНТЯБРЬ 2026\n"
        "\n"
        "🔎 Работа • Сервер B\n"
        "\n"
        "📭 Данных нет."
    )


def test_source_facade_is_category_local(conn: sqlite3.Connection) -> None:
    """A source under a DIFFERENT category is invisible to this query."""
    persist_row(
        conn,
        transaction_date=date(2026, 9, 5),
        amount_usdt=Decimal(14),
        category="Сервисы",
        source="Проект A",
    )

    # «Проект A» exists in September 2026, but only under Сервисы.
    assert get_monthly_source_summary_text(
        conn, year=2026, month=9, category="Работа", source="Проект A"
    ) == (
        "💰 FINANCE | СЕНТЯБРЬ 2026\n"
        "\n"
        "🔎 Работа • Проект A\n"
        "\n"
        "📭 Данных нет."
    )
    assert "Доход: 14 USDT" in get_monthly_source_summary_text(
        conn, year=2026, month=9, category="Сервисы", source="Проект A"
    )


def test_facades_delegates_to_build_monthly_report() -> None:
    """The G2 facades compose the accepted E1 builder, nothing else."""
    for facade in (
        integration_module.get_monthly_category_summary_text,
        integration_module.get_monthly_source_summary_text,
    ):
        source = inspect.getsource(facade)
        assert "build_monthly_report" in source
        # No separate transaction-list query and no SQL of its own.
        assert "list_transactions" not in source
        assert "SELECT" not in source
        assert ".execute" not in source
        assert "FinanceConfig" not in source


@pytest.mark.parametrize(
    ("year", "month"),
    [
        (2026, 0),
        (2026, 13),
        (0, 9),
        (10000, 9),
        (2026, "9"),
        (2026, None),
        (2026, True),
        ("2026", 9),
    ],
)
def test_g2_year_month_validation_propagates_from_e1(
    year: Any, month: Any, conn: sqlite3.Connection
) -> None:
    with pytest.raises((TypeError, ValueError)):
        get_monthly_category_summary_text(
            conn, year=year, month=month, category="Работа"
        )
    with pytest.raises((TypeError, ValueError)):
        get_monthly_source_summary_text(
            conn, year=year, month=month, category="Работа", source="Проект A",
        )


@pytest.mark.parametrize("blank_category", ["", " ", "\t"])
def test_g2_blank_category_propagates_validation(
    blank_category: str, conn: sqlite3.Connection
) -> None:
    with pytest.raises(ValueError):
        get_monthly_category_summary_text(
            conn, year=2026, month=9, category=blank_category
        )
    with pytest.raises(ValueError):
        get_monthly_source_summary_text(
            conn, year=2026, month=9, category=blank_category, source="Проект A"
        )


@pytest.mark.parametrize("blank_source", ["", " ", "\t"])
def test_g2_blank_source_propagates_validation(
    blank_source: str, conn: sqlite3.Connection
) -> None:
    with pytest.raises(ValueError):
        get_monthly_source_summary_text(
            conn, year=2026, month=9, category="Работа", source=blank_source
        )


def test_g2_blank_source_rejected_even_when_category_missing(
    conn: sqlite3.Connection,
) -> None:
    """Source validation does not depend on the category existing."""
    with pytest.raises(ValueError):
        get_monthly_source_summary_text(
            conn, year=2026, month=9, category="Инфраструктура", source="  "
        )


@pytest.mark.parametrize("bad_category", [None, 5, b"bytes-label", object()])
def test_g2_non_string_category_propagates_validation(
    bad_category: Any, conn: sqlite3.Connection
) -> None:
    with pytest.raises(TypeError):
        get_monthly_category_summary_text(
            conn, year=2026, month=9, category=bad_category
        )


def test_g2_corrupt_row_propagates_repository_data_error(
    conn: sqlite3.Connection,
) -> None:
    persist_canonical_september(conn)
    conn.execute(
        "INSERT INTO transactions ("
        " direction, amount_usdt, category, source, comment,"
        " transaction_date, created_at, updated_at, status, deleted_at,"
        " chat_id, message_thread_id, message_id, update_id"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "income",
            "not-a-decimal",
            "Работа",
            "Проект A",
            None,
            "2026-09-04",
            FIXED_CREATED_AT.isoformat(),
            FIXED_CREATED_AT.isoformat(),
            "active",
            None,
            -100200,
            7,
            999,
            10999,
        ),
    )
    conn.commit()

    with pytest.raises(RepositoryDataError) as exc_info:
        get_monthly_category_summary_text(
            conn, year=2026, month=9, category="Работа"
        )
    assert exc_info.value.__cause__ is not None

    with pytest.raises(RepositoryDataError):
        get_monthly_source_summary_text(
            conn, year=2026, month=9, category="Работа", source="Проект A"
        )


def test_g2_facades_are_read_only(conn: sqlite3.Connection) -> None:
    """No facade writes rows or touches the caller's transaction state."""
    persist_canonical_september(conn)

    conn.execute("BEGIN")
    assert conn.in_transaction is True
    before = conn.execute("SELECT * FROM transactions").fetchall()
    before_updates = conn.execute("SELECT * FROM processed_updates").fetchall()

    get_monthly_category_summary_text(conn, year=2026, month=9, category="Работа")
    get_monthly_source_summary_text(
        conn, year=2026, month=9, category="Работа", source="Проект A"
    )

    assert conn.in_transaction is True
    assert conn.execute("SELECT * FROM transactions").fetchall() == before
    assert (
        conn.execute("SELECT * FROM processed_updates").fetchall() == before_updates
    )
    conn.rollback()


def test_g2_soft_deleted_transactions_are_invisible(conn: sqlite3.Connection) -> None:
    """D1 ACTIVE-only visibility flows through E1 into the summaries."""
    persisted = persist_row(
        conn,
        transaction_date=date(2026, 9, 5),
        amount_usdt=Decimal(25),
        category="Работа",
        source="Проект A",
    )
    assert "Доход: 25 USDT" in get_monthly_category_summary_text(
        conn, year=2026, month=9, category="Работа"
    )

    soft_delete_transaction(
        conn, persisted.transaction_id or "", deleted_at=FIXED_PROCESSED_AT
    )

    # With no remaining ACTIVE transactions, E1 holds no category for
    # the month: the label no longer exists in the selected month, so
    # the explicit no-data state -- not a manufactured zero report --
    # is the authoritative answer.
    assert get_monthly_category_summary_text(
        conn, year=2026, month=9, category="Работа"
    ) == (
        "💰 FINANCE | СЕНТЯБРЬ 2026\n"
        "\n"
        "🔎 Категория: Работа\n"
        "\n"
        "📭 Данных нет."
    )


def test_g2_only_selected_month_contributes(conn: sqlite3.Connection) -> None:
    persist_row(
        conn,
        transaction_date=date(2026, 8, 20),
        amount_usdt=Decimal(25),
        category="Работа",
        source="Проект A",
    )

    september = get_monthly_category_summary_text(
        conn, year=2026, month=9, category="Работа"
    )
    assert "📭 Данных нет." in september

    august = get_monthly_category_summary_text(
        conn, year=2026, month=8, category="Работа"
    )
    assert "📈 Доход: 25 USDT" in august


# ---------------------------------------------------------------------------
# public API and module hygiene
# ---------------------------------------------------------------------------


def test_filtered_summary_public_api_is_small_and_deliberate() -> None:
    assert filtered_summary_module.__all__ == [
        "select_category_source",
        "select_monthly_category",
    ]
    for name in filtered_summary_module.__all__:
        assert callable(getattr(filtered_summary_module, name))
        assert not name.startswith("_")


def test_package_exports_g2_api() -> None:
    assert hermes_finance.select_monthly_category is select_monthly_category
    assert hermes_finance.select_category_source is select_category_source
    assert (
        hermes_finance.render_monthly_category_summary
        is render_monthly_category_summary
    )
    assert hermes_finance.render_monthly_source_summary is render_monthly_source_summary
    assert (
        hermes_finance.get_monthly_category_summary_text
        is get_monthly_category_summary_text
    )
    assert (
        hermes_finance.get_monthly_source_summary_text
        is get_monthly_source_summary_text
    )
    for name in (
        "select_monthly_category",
        "select_category_source",
        "render_monthly_category_summary",
        "render_monthly_source_summary",
        "get_monthly_category_summary_text",
        "get_monthly_source_summary_text",
    ):
        assert name in hermes_finance.__all__


@pytest.mark.parametrize(
    "forbidden",
    [
        "sqlite3",
        "SELECT",
        "INSERT",
        "UPDATE",
        "DELETE",
        "PRAGMA",
        ".execute",
        "datetime.now",
        "datetime.today",
        "utcnow",
        "today(",
        "time.time",
        "time.monotonic",
        "perf_counter",
        "import os",
        "import time",
        "FinanceConfig",
        "open_database",
    ],
)
def test_filtered_summary_source_avoids_forbidden_operations(forbidden: str) -> None:
    """Selection is pure: no SQL, no connection, no clock, no config."""
    assert forbidden not in FILTERED_SOURCE


@pytest.mark.parametrize(
    "forbidden",
    [
        ".casefold(",
        ".lower(",
        ".upper(",
        ".strip(",
        ".lstrip(",
        ".rstrip(",
        ".startswith(",
        ".endswith(",
        "Decimal(",
        "sum(",
        "round(",
        "float(",
        "quantize(",
    ],
)
def test_filtered_summary_source_avoids_matching_and_money_logic(
    forbidden: str,
) -> None:
    """No competing matching policy and no money arithmetic exist.

    The only label handling is the accepted required-text normalisation
    delegated to the domain layer; the deliberate "no casefolding, no
    fuzzy matching" prose in the module docstring documents the policy,
    while these pins prove no matching or arithmetic code exists.
    """
    assert forbidden not in FILTERED_SOURCE
