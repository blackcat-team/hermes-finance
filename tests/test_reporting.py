"""Tests for the deterministic monthly reporting aggregation (stage E1).

All tests are deterministic: no network, no wall-clock time, no
randomness, no environment access. Filesystem mutation happens only
inside in-memory SQLite databases (``:memory:``).

Money exactness is proven without floats: values are compared with
exact ``Decimal`` equality, and high-precision sums are additionally
verified through exact :class:`fractions.Fraction` conversion, which
is independent of any decimal-context precision setting.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from collections.abc import Iterator
from datetime import UTC, date, datetime
from decimal import Decimal, getcontext, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

import pytest

import hermes_finance
import hermes_finance.reporting as reporting_module
from hermes_finance import (
    CategoryReport,
    Direction,
    FinanceConfig,
    IngestDisposition,
    MonthlyFinanceReport,
    SourceReport,
    TelegramMessageRef,
    Transaction,
    TransactionStatus,
    build_monthly_report,
    edit_transaction,
    ingest_transaction,
    open_database,
    soft_delete_transaction,
)
from hermes_finance.repository import RepositoryDataError, persist_transaction

REPORTING_SOURCE: str = Path(reporting_module.__file__).read_text(encoding="utf-8")

FIXED_TZ = UTC
FIXED_CREATED_AT = datetime(2026, 9, 1, 10, 0, 0, tzinfo=FIXED_TZ)
FIXED_UPDATED_AT = datetime(2026, 9, 1, 11, 30, 0, tzinfo=FIXED_TZ)
FIXED_PROCESSED_AT = datetime(2026, 9, 1, 12, 15, 0, tzinfo=FIXED_TZ)
FIXED_DELETED_AT = datetime(2026, 9, 1, 13, 0, 0, tzinfo=FIXED_TZ)

ZERO: Final[Decimal] = Decimal(0)


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
        "transaction_date": date(2026, 9, 1),
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


def insert_raw_row(
    connection: sqlite3.Connection,
    *,
    message_id: int,
    transaction_date: str,
    amount_usdt: Any = "25",
    direction: str = "income",
) -> None:
    """Insert one raw ACTIVE transactions row, bypassing the domain layer.

    Deliberately does not commit: the caller owns the transaction state.
    """
    connection.execute(
        "INSERT INTO transactions ("
        " direction, amount_usdt, category, source, comment,"
        " transaction_date, created_at, updated_at, status, deleted_at,"
        " chat_id, message_thread_id, message_id, update_id"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            direction,
            amount_usdt,
            "Работа",
            "Проект A",
            None,
            transaction_date,
            "2026-09-01T10:00:00+00:00",
            "2026-09-01T10:00:00+00:00",
            "active",
            None,
            -100200,
            7,
            message_id,
            20_000 + message_id,
        ),
    )


def categories_of(report: MonthlyFinanceReport) -> list[str]:
    """The category names of a report, in report order."""
    return [category.category for category in report.categories]


def sources_of(category: CategoryReport) -> list[str]:
    """The source names of a category report, in report order."""
    return [source.source for source in category.sources]


def category_by_name(report: MonthlyFinanceReport, name: str) -> CategoryReport:
    """The CategoryReport with the exact given category name."""
    for category in report.categories:
        if category.category == name:
            return category
    raise AssertionError(f"category {name!r} not found in report")


def source_by_name(category: CategoryReport, name: str) -> SourceReport:
    """The SourceReport with the exact given source name."""
    for source in category.sources:
        if source.source == name:
            return source
    raise AssertionError(f"source {name!r} not found in category {category.category!r}")


# ---------------------------------------------------------------------------
# empty month
# ---------------------------------------------------------------------------


def test_empty_month_returns_exact_zero_report(conn: sqlite3.Connection) -> None:
    report = build_monthly_report(conn, year=2026, month=5)

    assert report.year == 2026
    assert report.month == 5
    assert report.income_usdt == ZERO
    assert report.expense_usdt == ZERO
    assert report.net_usdt == ZERO
    assert report.transaction_count == 0
    assert report.categories == ()
    assert report is not None


def test_empty_month_after_deleting_only_transaction(conn: sqlite3.Connection) -> None:
    persisted = persist_row(conn, transaction_date=date(2026, 5, 10))
    soft_delete_transaction(conn, persisted.transaction_id or "", deleted_at=FIXED_DELETED_AT)

    report = build_monthly_report(conn, year=2026, month=5)

    assert report.income_usdt == ZERO
    assert report.expense_usdt == ZERO
    assert report.net_usdt == ZERO
    assert report.transaction_count == 0
    assert report.categories == ()


# ---------------------------------------------------------------------------
# mixed directions: month totals, category/source breakdown, invariants
# ---------------------------------------------------------------------------


def persist_mixed_month(connection: sqlite3.Connection, year: int, month: int) -> None:
    """Persist the canonical mixed-direction month in scrambled order.

    Deliberately scrambled insertion order: Инфраструктура, Работа/Проект A,
    Инфраструктура/Сервер A, Сервисы, Работа/Проект B,
    Инфраструктура/Хостинг. The report ordering must not follow it.
    """
    persist_row(
        connection,
        transaction_date=date(year, month, 3),
        direction=Direction.EXPENSE,
        amount_usdt=Decimal(10),
        category="Инфраструктура",
        source="Сервер B",
    )
    persist_row(
        connection,
        transaction_date=date(year, month, 1),
        direction=Direction.INCOME,
        amount_usdt=Decimal(25),
        category="Работа",
        source="Проект A",
    )
    persist_row(
        connection,
        transaction_date=date(year, month, 5),
        direction=Direction.EXPENSE,
        amount_usdt=Decimal(1),
        category="Инфраструктура",
        source="Сервер A",
    )
    persist_row(
        connection,
        transaction_date=date(year, month, 7),
        direction=Direction.INCOME,
        amount_usdt=Decimal(5),
        category="Сервисы",
        source="Подписка",
    )
    persist_row(
        connection,
        transaction_date=date(year, month, 9),
        direction=Direction.INCOME,
        amount_usdt=Decimal(30),
        category="Работа",
        source="Проект B",
    )
    persist_row(
        connection,
        transaction_date=date(year, month, 11),
        direction=Direction.EXPENSE,
        amount_usdt=Decimal(10),
        category="Инфраструктура",
        source="Хостинг",
    )


def test_mixed_directions_month_totals(conn: sqlite3.Connection) -> None:
    persist_mixed_month(conn, 2026, 9)

    report = build_monthly_report(conn, year=2026, month=9)

    assert report.year == 2026
    assert report.month == 9
    assert report.income_usdt == Decimal(60)
    assert report.expense_usdt == Decimal(21)
    assert report.net_usdt == Decimal(39)
    assert report.transaction_count == 6
    # Expense stays a positive magnitude; only net subtracts.
    assert report.expense_usdt == Decimal(21)
    assert report.expense_usdt != Decimal(-21)


def test_mixed_directions_category_and_source_breakdown(conn: sqlite3.Connection) -> None:
    persist_mixed_month(conn, 2026, 9)

    report = build_monthly_report(conn, year=2026, month=9)

    assert categories_of(report) == ["Инфраструктура", "Работа", "Сервисы"]

    infrastructure = category_by_name(report, "Инфраструктура")
    assert infrastructure.income_usdt == ZERO
    assert infrastructure.expense_usdt == Decimal(21)
    assert infrastructure.net_usdt == Decimal(-21)
    assert infrastructure.transaction_count == 3
    # Lexical order: "Сервер A" < "Сервер B" < "Хостинг".
    assert sources_of(infrastructure) == ["Сервер A", "Сервер B", "Хостинг"]
    server_a = source_by_name(infrastructure, "Сервер A")
    assert server_a.income_usdt == ZERO
    assert server_a.expense_usdt == Decimal(1)
    assert server_a.net_usdt == Decimal(-1)
    assert server_a.transaction_count == 1
    server_b = source_by_name(infrastructure, "Сервер B")
    assert server_b.income_usdt == ZERO
    assert server_b.expense_usdt == Decimal(10)
    assert server_b.net_usdt == Decimal(-10)
    assert server_b.transaction_count == 1
    hosting = source_by_name(infrastructure, "Хостинг")
    assert hosting.income_usdt == ZERO
    assert hosting.expense_usdt == Decimal(10)
    assert hosting.net_usdt == Decimal(-10)
    assert hosting.transaction_count == 1

    work = category_by_name(report, "Работа")
    assert work.income_usdt == Decimal(55)
    assert work.expense_usdt == ZERO
    assert work.net_usdt == Decimal(55)
    assert work.transaction_count == 2
    # Lexical order: "Проект A" < "Проект B".
    assert sources_of(work) == ["Проект A", "Проект B"]
    project_a = source_by_name(work, "Проект A")
    assert project_a.income_usdt == Decimal(25)
    assert project_a.net_usdt == Decimal(25)
    assert project_a.transaction_count == 1
    project_b = source_by_name(work, "Проект B")
    assert project_b.income_usdt == Decimal(30)
    assert project_b.net_usdt == Decimal(30)
    assert project_b.transaction_count == 1

    services = category_by_name(report, "Сервисы")
    assert services.income_usdt == Decimal(5)
    assert services.expense_usdt == ZERO
    assert services.net_usdt == Decimal(5)
    assert services.transaction_count == 1
    assert sources_of(services) == ["Подписка"]
    subscription = source_by_name(services, "Подписка")
    assert subscription.income_usdt == Decimal(5)
    assert subscription.expense_usdt == ZERO
    assert subscription.net_usdt == Decimal(5)
    assert subscription.transaction_count == 1


def test_month_and_category_invariants(conn: sqlite3.Connection) -> None:
    persist_mixed_month(conn, 2026, 9)

    report = build_monthly_report(conn, year=2026, month=9)

    for category in report.categories:
        assert category.income_usdt == sum(
            (source.income_usdt for source in category.sources), ZERO
        )
        assert category.expense_usdt == sum(
            (source.expense_usdt for source in category.sources), ZERO
        )
        assert category.net_usdt == category.income_usdt - category.expense_usdt
        assert category.transaction_count == sum(
            source.transaction_count for source in category.sources
        )

    assert report.income_usdt == sum((c.income_usdt for c in report.categories), ZERO)
    assert report.expense_usdt == sum((c.expense_usdt for c in report.categories), ZERO)
    assert report.net_usdt == report.income_usdt - report.expense_usdt
    assert report.transaction_count == sum(c.transaction_count for c in report.categories)


# ---------------------------------------------------------------------------
# exact Decimal precision
# ---------------------------------------------------------------------------


def test_exact_precision_high_digit_amounts(conn: sqlite3.Connection) -> None:
    high = Decimal("1.2345678901234567890123456789")
    tiny = Decimal("0.000001")
    huge = Decimal("999999999999999999.999999999999999999")
    persist_row(
        conn,
        transaction_date=date(2026, 9, 1),
        amount_usdt=high,
        category="Высокая точность",
        source="S1",
    )
    persist_row(
        conn,
        transaction_date=date(2026, 9, 2),
        amount_usdt=high,
        category="Высокая точность",
        source="S1",
    )
    persist_row(
        conn,
        transaction_date=date(2026, 9, 3),
        amount_usdt=tiny,
        category="Высокая точность",
        source="S2",
    )
    persist_row(
        conn,
        transaction_date=date(2026, 9, 4),
        direction=Direction.EXPENSE,
        amount_usdt=huge,
        category="Крупная",
        source="B1",
    )

    report = build_monthly_report(conn, year=2026, month=9)

    expected_income = 2 * Fraction(high) + Fraction(tiny)
    expected_expense = Fraction(huge)
    assert report.transaction_count == 4
    assert Fraction(report.income_usdt) == expected_income
    assert Fraction(report.expense_usdt) == expected_expense
    assert Fraction(report.net_usdt) == expected_income - expected_expense

    precision_category = category_by_name(report, "Высокая точность")
    assert Fraction(precision_category.income_usdt) == expected_income
    s1 = source_by_name(precision_category, "S1")
    assert Fraction(s1.income_usdt) == 2 * Fraction(high)
    s2 = source_by_name(precision_category, "S2")
    assert Fraction(s2.income_usdt) == Fraction(tiny)

    big_category = category_by_name(report, "Крупная")
    assert Fraction(big_category.expense_usdt) == expected_expense
    assert Fraction(big_category.net_usdt) == -expected_expense


def test_exact_precision_small_and_trailing_zero_amounts(conn: sqlite3.Connection) -> None:
    persist_row(
        conn,
        transaction_date=date(2026, 9, 1),
        amount_usdt=Decimal("123.4500"),
        category="Работа",
        source="Проект A",
    )
    persist_row(
        conn,
        transaction_date=date(2026, 9, 2),
        amount_usdt=Decimal("0.000001"),
        category="Работа",
        source="Проект A",
    )
    persist_row(
        conn,
        transaction_date=date(2026, 9, 3),
        direction=Direction.EXPENSE,
        amount_usdt=Decimal("0.000002"),
        category="Инфраструктура",
        source="Хостинг",
    )

    report = build_monthly_report(conn, year=2026, month=9)

    assert report.income_usdt == Decimal("123.450001")
    assert report.expense_usdt == Decimal("0.000002")
    assert report.net_usdt == Decimal("123.449999")
    assert report.transaction_count == 3
    # No quantisation to a fixed number of decimal places.
    assert report.income_usdt.as_tuple().exponent == -6
    project_a = source_by_name(category_by_name(report, "Работа"), "Проект A")
    assert project_a.income_usdt == Decimal("123.450001")
    assert project_a.transaction_count == 2


# ---------------------------------------------------------------------------
# grouping semantics
# ---------------------------------------------------------------------------


def test_same_source_in_different_categories_is_not_merged(conn: sqlite3.Connection) -> None:
    persist_row(
        conn,
        transaction_date=date(2026, 9, 1),
        direction=Direction.INCOME,
        amount_usdt=Decimal(10),
        category="Работа",
        source="Shared",
    )
    persist_row(
        conn,
        transaction_date=date(2026, 9, 2),
        direction=Direction.EXPENSE,
        amount_usdt=Decimal(3),
        category="Инфраструктура",
        source="Shared",
    )

    report = build_monthly_report(conn, year=2026, month=9)

    assert report.income_usdt == Decimal(10)
    assert report.expense_usdt == Decimal(3)
    assert report.net_usdt == Decimal(7)
    assert report.transaction_count == 2
    assert categories_of(report) == ["Инфраструктура", "Работа"]

    work = category_by_name(report, "Работа")
    assert len(work.sources) == 1
    work_shared = source_by_name(work, "Shared")
    assert work_shared.income_usdt == Decimal(10)
    assert work_shared.expense_usdt == ZERO
    assert work_shared.net_usdt == Decimal(10)
    assert work_shared.transaction_count == 1

    infrastructure = category_by_name(report, "Инфраструктура")
    assert len(infrastructure.sources) == 1
    infra_shared = source_by_name(infrastructure, "Shared")
    assert infra_shared.income_usdt == ZERO
    assert infra_shared.expense_usdt == Decimal(3)
    assert infra_shared.net_usdt == Decimal(-3)
    assert infra_shared.transaction_count == 1


def test_repeated_category_source_accumulates_into_one_source_report(
    conn: sqlite3.Connection,
) -> None:
    persist_row(
        conn,
        transaction_date=date(2026, 9, 1),
        amount_usdt=Decimal(10),
        category="Работа",
        source="Проект A",
    )
    persist_row(
        conn,
        transaction_date=date(2026, 9, 2),
        amount_usdt=Decimal(15),
        category="Работа",
        source="Проект A",
    )
    persist_row(
        conn,
        transaction_date=date(2026, 9, 3),
        direction=Direction.EXPENSE,
        amount_usdt=Decimal(2),
        category="Работа",
        source="Проект A",
    )

    report = build_monthly_report(conn, year=2026, month=9)

    assert report.transaction_count == 3
    work = category_by_name(report, "Работа")
    assert len(work.sources) == 1
    project_a = source_by_name(work, "Проект A")
    assert project_a.income_usdt == Decimal(25)
    assert project_a.expense_usdt == Decimal(2)
    assert project_a.net_usdt == Decimal(23)
    assert project_a.transaction_count == 3
    assert work.income_usdt == Decimal(25)
    assert work.expense_usdt == Decimal(2)
    assert work.net_usdt == Decimal(23)
    assert work.transaction_count == 3


def test_category_and_source_with_both_directions(conn: sqlite3.Connection) -> None:
    persist_row(
        conn,
        transaction_date=date(2026, 9, 1),
        amount_usdt=Decimal(100),
        category="Работа",
        source="Проект B",
    )
    persist_row(
        conn,
        transaction_date=date(2026, 9, 2),
        direction=Direction.EXPENSE,
        amount_usdt=Decimal(20),
        category="Работа",
        source="Проект B",
    )

    report = build_monthly_report(conn, year=2026, month=9)

    work = category_by_name(report, "Работа")
    assert work.income_usdt == Decimal(100)
    assert work.expense_usdt == Decimal(20)
    assert work.net_usdt == Decimal(80)
    assert work.transaction_count == 2
    project_b = source_by_name(work, "Проект B")
    assert project_b.income_usdt == Decimal(100)
    assert project_b.expense_usdt == Decimal(20)
    assert project_b.net_usdt == Decimal(80)
    assert project_b.transaction_count == 2


def test_category_and_source_strings_preserved_exactly(conn: sqlite3.Connection) -> None:
    # Different case is a different exact persisted key: no casefolding.
    persist_row(
        conn,
        transaction_date=date(2026, 9, 1),
        amount_usdt=Decimal(1),
        category="Alpha",
        source="Beta",
    )
    persist_row(
        conn,
        transaction_date=date(2026, 9, 2),
        amount_usdt=Decimal(2),
        category="alpha",
        source="beta",
    )
    persist_row(
        conn,
        transaction_date=date(2026, 9, 3),
        amount_usdt=Decimal(4),
        category="alpha",
        source="Beta",
    )

    report = build_monthly_report(conn, year=2026, month=9)

    # ASCII code-point order: "Alpha" < "alpha".
    assert categories_of(report) == ["Alpha", "alpha"]
    assert source_by_name(category_by_name(report, "Alpha"), "Beta").transaction_count == 1
    alpha = category_by_name(report, "alpha")
    assert sources_of(alpha) == ["Beta", "beta"]
    assert source_by_name(alpha, "Beta").transaction_count == 1
    assert source_by_name(alpha, "beta").transaction_count == 1


def test_lexical_ordering_independent_of_insertion_order(conn: sqlite3.Connection) -> None:
    # Deliberately reverse-lexically scrambled insertion.
    for index, name in enumerate(["delta", "Charlie", "alpha", "Echo", "bravo"]):
        persist_row(
            conn,
            transaction_date=date(2026, 9, 1 + index),
            amount_usdt=Decimal(1),
            category=name,
            source="Zulu",
        )
    for index, name in enumerate(["yankee", "xray", "Whiskey", "victor", "Uniform"]):
        persist_row(
            conn,
            transaction_date=date(2026, 9, 10 + index),
            amount_usdt=Decimal(1),
            category="alpha",
            source=name,
        )

    report = build_monthly_report(conn, year=2026, month=9)

    assert categories_of(report) == ["Charlie", "Echo", "alpha", "bravo", "delta"]
    alpha = category_by_name(report, "alpha")
    assert sources_of(alpha) == ["Uniform", "Whiskey", "Zulu", "victor", "xray", "yankee"]


# ---------------------------------------------------------------------------
# report model immutability
# ---------------------------------------------------------------------------


def test_report_models_are_frozen_with_slots_and_tuples(conn: sqlite3.Connection) -> None:
    persist_mixed_month(conn, 2026, 9)
    report = build_monthly_report(conn, year=2026, month=9)

    assert dataclasses.is_dataclass(report)
    assert isinstance(report.categories, tuple)
    for category in report.categories:
        assert isinstance(category.sources, tuple)
        for source in category.sources:
            assert not hasattr(source, "__dict__")
            with pytest.raises(dataclasses.FrozenInstanceError):
                source.income_usdt = Decimal(1)  # type: ignore
        assert not hasattr(category, "__dict__")
        with pytest.raises(dataclasses.FrozenInstanceError):
            category.sources = ()  # type: ignore
    assert not hasattr(report, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.year = 2027  # type: ignore
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.categories = ()  # type: ignore


# ---------------------------------------------------------------------------
# month boundaries (through D1)
# ---------------------------------------------------------------------------


def test_leap_february_boundaries(conn: sqlite3.Connection) -> None:
    persist_row(conn, transaction_date=date(2028, 1, 31), amount_usdt=Decimal(1))
    persist_row(conn, transaction_date=date(2028, 2, 28), amount_usdt=Decimal(2))
    persist_row(conn, transaction_date=date(2028, 2, 29), amount_usdt=Decimal(4))
    persist_row(conn, transaction_date=date(2028, 3, 1), amount_usdt=Decimal(8))

    report = build_monthly_report(conn, year=2028, month=2)

    assert report.income_usdt == Decimal(6)
    assert report.transaction_count == 2
    assert build_monthly_report(conn, year=2028, month=1).transaction_count == 1
    assert build_monthly_report(conn, year=2028, month=3).transaction_count == 1


def test_december_january_rollover(conn: sqlite3.Connection) -> None:
    persist_row(conn, transaction_date=date(2026, 12, 31), amount_usdt=Decimal(5))
    persist_row(conn, transaction_date=date(2027, 1, 1), amount_usdt=Decimal(7))

    december = build_monthly_report(conn, year=2026, month=12)
    january = build_monthly_report(conn, year=2027, month=1)

    assert december.income_usdt == Decimal(5)
    assert december.transaction_count == 1
    assert december.year == 2026
    assert december.month == 12
    assert january.income_usdt == Decimal(7)
    assert january.transaction_count == 1
    assert january.year == 2027
    assert january.month == 1


def test_year_9999_december_supported(conn: sqlite3.Connection) -> None:
    persist_row(conn, transaction_date=date(9999, 12, 31), amount_usdt=Decimal(9))

    report = build_monthly_report(conn, year=9999, month=12)

    assert report.income_usdt == Decimal(9)
    assert report.transaction_count == 1
    assert report.year == 9999
    assert report.month == 12
    assert build_monthly_report(conn, year=9999, month=11).transaction_count == 0


# ---------------------------------------------------------------------------
# D2 integration: soft delete and edit
# ---------------------------------------------------------------------------


def test_soft_deleted_transactions_contribute_nothing(conn: sqlite3.Connection) -> None:
    persisted = persist_row(
        conn,
        transaction_date=date(2026, 9, 1),
        amount_usdt=Decimal(25),
        category="Работа",
        source="Проект A",
    )
    persist_mixed_month(conn, 2026, 9)

    before = build_monthly_report(conn, year=2026, month=9)
    assert before.income_usdt == Decimal(85)
    assert before.transaction_count == 7

    soft_delete_transaction(conn, persisted.transaction_id or "", deleted_at=FIXED_DELETED_AT)

    after = build_monthly_report(conn, year=2026, month=9)
    assert after.income_usdt == Decimal(60)
    assert after.expense_usdt == Decimal(21)
    assert after.net_usdt == Decimal(39)
    assert after.transaction_count == 6
    assert categories_of(after) == ["Инфраструктура", "Работа", "Сервисы"]


def test_edit_moves_contribution_between_months(conn: sqlite3.Connection) -> None:
    persisted = persist_row(
        conn,
        transaction_date=date(2026, 1, 15),
        amount_usdt=Decimal(25),
        category="Работа",
        source="Проект A",
    )

    january_before = build_monthly_report(conn, year=2026, month=1)
    assert january_before.income_usdt == Decimal(25)
    assert january_before.transaction_count == 1

    edit_transaction(
        conn,
        persisted.transaction_id or "",
        direction=Direction.EXPENSE,
        amount_usdt=Decimal(40),
        category="Инфраструктура",
        source="Сервер B",
        comment=None,
        transaction_date=date(2026, 2, 10),
        updated_at=FIXED_UPDATED_AT,
    )

    january_after = build_monthly_report(conn, year=2026, month=1)
    assert january_after.income_usdt == ZERO
    assert january_after.expense_usdt == ZERO
    assert january_after.net_usdt == ZERO
    assert january_after.transaction_count == 0
    assert january_after.categories == ()

    february = build_monthly_report(conn, year=2026, month=2)
    assert february.income_usdt == ZERO
    assert february.expense_usdt == Decimal(40)
    assert february.net_usdt == Decimal(-40)
    assert february.transaction_count == 1
    assert categories_of(february) == ["Инфраструктура"]
    infrastructure = category_by_name(february, "Инфраструктура")
    assert sources_of(infrastructure) == ["Сервер B"]
    server_b = source_by_name(infrastructure, "Сервер B")
    assert server_b.expense_usdt == Decimal(40)
    assert server_b.transaction_count == 1


# ---------------------------------------------------------------------------
# idempotent ingest
# ---------------------------------------------------------------------------


def test_duplicate_deliveries_count_exactly_once(conn: sqlite3.Connection) -> None:
    transaction = make_transaction(
        transaction_date=date(2026, 9, 1), amount_usdt=Decimal(10)
    )

    first = ingest_transaction(
        conn, transaction, make_ref(message_id=100, update_id=1000), processed_at=FIXED_PROCESSED_AT
    )
    assert first.disposition is IngestDisposition.CREATED

    second = ingest_transaction(
        conn, transaction, make_ref(message_id=100, update_id=1001), processed_at=FIXED_PROCESSED_AT
    )
    assert second.disposition is IngestDisposition.DUPLICATE_MESSAGE

    third = ingest_transaction(
        conn, transaction, make_ref(message_id=100, update_id=1000), processed_at=FIXED_PROCESSED_AT
    )
    assert third.disposition is IngestDisposition.DUPLICATE_UPDATE

    report = build_monthly_report(conn, year=2026, month=9)

    assert report.income_usdt == Decimal(10)
    assert report.expense_usdt == ZERO
    assert report.net_usdt == Decimal(10)
    assert report.transaction_count == 1
    assert categories_of(report) == ["Работа"]
    work = category_by_name(report, "Работа")
    assert len(work.sources) == 1
    assert work.transaction_count == 1
    assert source_by_name(work, "Проект A").transaction_count == 1


# ---------------------------------------------------------------------------
# read-only semantics: caller transaction, corrupt-row propagation
# ---------------------------------------------------------------------------


def test_caller_pending_transaction_visible_and_never_committed(
    conn: sqlite3.Connection,
) -> None:
    conn.execute("BEGIN")
    insert_raw_row(conn, message_id=500, transaction_date="2026-09-05", amount_usdt="7")

    pending_report = build_monthly_report(conn, year=2026, month=9)

    # The report may see caller-pending state exactly as D1 does...
    assert pending_report.income_usdt == Decimal(7)
    assert pending_report.transaction_count == 1
    # ...and the caller transaction is still active afterwards.
    assert conn.in_transaction

    conn.rollback()

    rolled_back_report = build_monthly_report(conn, year=2026, month=9)
    assert rolled_back_report.income_usdt == ZERO
    assert rolled_back_report.transaction_count == 0
    assert rolled_back_report.categories == ()


def test_corrupt_active_row_propagates_repository_data_error(
    conn: sqlite3.Connection,
) -> None:
    persist_row(conn, transaction_date=date(2026, 9, 1), amount_usdt=Decimal(5))
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
            "2026-09-02",
            "2026-09-01T10:00:00+00:00",
            "2026-09-01T10:00:00+00:00",
            "active",
            None,
            -100200,
            7,
            501,
            20501,
        ),
    )
    conn.commit()

    with pytest.raises(RepositoryDataError) as exc_info:
        build_monthly_report(conn, year=2026, month=9)

    assert exc_info.value.__cause__ is not None


# ---------------------------------------------------------------------------
# input validation (delegated to D1) and connection contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("year", "month"),
    [
        (True, 9),
        (1.5, 9),
        ("2026", 9),
        (None, 9),
        (0, 9),
        (10000, 9),
        (2026, True),
        (2026, 1.5),
        (2026, "9"),
        (2026, None),
        (2026, 0),
        (2026, 13),
    ],
)
def test_invalid_year_or_month_rejected_as_d1(
    conn: sqlite3.Connection, year: Any, month: Any
) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_monthly_report(conn, year=year, month=month)


@pytest.mark.parametrize("bad_connection", ["not-a-connection", None, 5, object()])
def test_non_connection_rejected(bad_connection: Any) -> None:
    with pytest.raises(TypeError):
        build_monthly_report(bad_connection, year=2026, month=9)


# ---------------------------------------------------------------------------
# public exports and module hygiene
# ---------------------------------------------------------------------------


def test_public_exports() -> None:
    assert hermes_finance.build_monthly_report is reporting_module.build_monthly_report
    assert hermes_finance.MonthlyFinanceReport is reporting_module.MonthlyFinanceReport
    assert hermes_finance.CategoryReport is reporting_module.CategoryReport
    assert hermes_finance.SourceReport is reporting_module.SourceReport
    for name in ("build_monthly_report", "MonthlyFinanceReport", "CategoryReport", "SourceReport"):
        assert name in hermes_finance.__all__
    # Private accumulation helpers stay private.
    for name in hermes_finance.__all__:
        assert not name.startswith("_")


def test_reporting_source_contains_no_sql_no_clock_no_mutation() -> None:
    for token in ("SELECT", "INSERT", "UPDATE", "DELETE", "PRAGMA", "BEGIN"):
        assert token not in REPORTING_SOURCE
    for token in ("execute(", "executemany(", "commit(", "rollback(", "cursor(", "connect("):
        assert token not in REPORTING_SOURCE
    for token in ("datetime", "import time", "time.time", "time.monotonic", "utcnow", "today("):
        assert token not in REPORTING_SOURCE
    # No finite hard-coded aggregation precision: exact accumulation must
    # not rely on any decimal-context precision mechanism at all.
    for token in ("localcontext", "MAX_PREC", "getcontext", "context.prec"):
        assert token not in REPORTING_SOURCE


# ---------------------------------------------------------------------------
# exact arbitrary-precision aggregation (QA remediation)
# ---------------------------------------------------------------------------


def test_exact_aggregation_under_low_ambient_context_precision(
    conn: sqlite3.Connection,
) -> None:
    """QA defect class: exact carry under a hostile low-precision context.

    36 nines plus 1 is exactly 10**36: the exact sum needs 37
    significant digits, so ordinary contextual Decimal addition under
    ``prec=3`` would round it. The report must stay exact.
    """
    with localcontext() as context:
        context.prec = 3
        persist_row(
            conn,
            transaction_date=date(2026, 9, 1),
            amount_usdt=Decimal("9" * 36),
            category="Работа",
            source="Проект B",
        )
        persist_row(
            conn,
            transaction_date=date(2026, 9, 2),
            amount_usdt=Decimal(1),
            category="Работа",
            source="Проект B",
        )
        report = build_monthly_report(conn, year=2026, month=9)
        # Still inside the hostile context: construction and comparison
        # of the expected value are exact and context-independent.
        assert report.income_usdt == Decimal(10**36)
        assert report.expense_usdt == ZERO
        assert report.net_usdt == Decimal(10**36)
        assert report.transaction_count == 2
        work = category_by_name(report, "Работа")
        assert work.income_usdt == Decimal(10**36)
        assert work.transaction_count == 2
        project_b = source_by_name(work, "Проект B")
        assert project_b.income_usdt == Decimal(10**36)
        assert project_b.net_usdt == Decimal(10**36)
        assert project_b.transaction_count == 2


def test_exact_aggregation_long_coefficients_with_carry(
    conn: sqlite3.Connection,
) -> None:
    """150-digit coefficients whose exact sum requires carrying."""
    a_text = "9" * 150
    b_text = "6" * 150
    expected = int(a_text) + int(b_text)

    with localcontext() as context:
        context.prec = 6
        persist_row(
            conn,
            transaction_date=date(2026, 9, 1),
            amount_usdt=Decimal(a_text),
            category="Работа",
            source="Проект B",
        )
        persist_row(
            conn,
            transaction_date=date(2026, 9, 2),
            amount_usdt=Decimal(b_text),
            category="Работа",
            source="Проект B",
        )
        report = build_monthly_report(conn, year=2026, month=9)

    assert report.income_usdt == Decimal(expected)
    assert report.net_usdt == Decimal(expected)
    project_b = source_by_name(category_by_name(report, "Работа"), "Проект B")
    assert project_b.income_usdt == Decimal(expected)
    assert project_b.transaction_count == 2


def test_exact_aggregation_large_exponent_gap(conn: sqlite3.Connection) -> None:
    """Exact alignment across a large but bounded exponent gap.

    10**30 plus 10**-30: the exact aggregate must retain BOTH the
    large-scale and the small-scale contribution. This catches
    implementations that preserve coefficient length but lose
    small-scale values.
    """
    big = Decimal("1" + "0" * 30)
    tiny = Decimal("0." + "0" * 29 + "1")
    expected = Fraction(big) + Fraction(tiny)

    with localcontext() as context:
        context.prec = 2
        persist_row(
            conn,
            transaction_date=date(2026, 9, 1),
            amount_usdt=big,
            category="Работа",
            source="Проект B",
        )
        persist_row(
            conn,
            transaction_date=date(2026, 9, 2),
            amount_usdt=tiny,
            category="Работа",
            source="Проект A",
        )
        report = build_monthly_report(conn, year=2026, month=9)

    assert Fraction(report.income_usdt) == expected
    assert report.transaction_count == 2
    work = category_by_name(report, "Работа")
    assert Fraction(work.income_usdt) == expected
    assert Fraction(source_by_name(work, "Проект B").income_usdt) == Fraction(big)
    assert Fraction(source_by_name(work, "Проект A").income_usdt) == Fraction(tiny)


def test_exact_net_retains_low_order_digits(conn: sqlite3.Connection) -> None:
    """Exact net with long coefficients and differing exponents.

    Income and expense both have long coefficients at different
    scales; the exact net must retain the low-order digits of both.
    Verified at month, category, and source level under a hostile
    low-precision ambient context.
    """
    income = Decimal("99999999999999999999999999999999999999999999999999.999999999999")
    expense = Decimal("123456789012345678901234567890.000000000001")
    expected_net = Fraction(income) - Fraction(expense)

    with localcontext() as context:
        context.prec = 6
        persist_row(
            conn,
            transaction_date=date(2026, 9, 1),
            direction=Direction.INCOME,
            amount_usdt=income,
            category="Работа",
            source="Проект B",
        )
        persist_row(
            conn,
            transaction_date=date(2026, 9, 2),
            direction=Direction.EXPENSE,
            amount_usdt=expense,
            category="Работа",
            source="Проект B",
        )
        report = build_monthly_report(conn, year=2026, month=9)

    assert Fraction(report.income_usdt) == Fraction(income)
    assert Fraction(report.expense_usdt) == Fraction(expense)
    assert Fraction(report.net_usdt) == expected_net
    assert report.transaction_count == 2

    work = category_by_name(report, "Работа")
    assert Fraction(work.income_usdt) == Fraction(income)
    assert Fraction(work.expense_usdt) == Fraction(expense)
    assert Fraction(work.net_usdt) == expected_net

    project_b = source_by_name(work, "Проект B")
    assert Fraction(project_b.income_usdt) == Fraction(income)
    assert Fraction(project_b.expense_usdt) == Fraction(expense)
    assert Fraction(project_b.net_usdt) == expected_net
    assert project_b.transaction_count == 2

    # Category and month invariants hold exactly.
    for category in report.categories:
        assert Fraction(category.net_usdt) == Fraction(category.income_usdt) - Fraction(
            category.expense_usdt
        )
    assert Fraction(report.net_usdt) == Fraction(report.income_usdt) - Fraction(
        report.expense_usdt
    )


def test_report_does_not_mutate_ambient_decimal_context(conn: sqlite3.Connection) -> None:
    """E1 must neither depend on nor permanently alter the caller context."""
    persist_mixed_month(conn, 2026, 9)
    default_prec = getcontext().prec

    with localcontext() as context:
        context.prec = 6
        report = build_monthly_report(conn, year=2026, month=9)
        # The caller's context precision is untouched right after the call.
        assert context.prec == 6
        assert getcontext().prec == 6

    # And the original context is intact after the caller block.
    assert getcontext().prec == default_prec

    # The report itself is still the exact mixed-month result.
    assert report.income_usdt == Decimal(60)
    assert report.expense_usdt == Decimal(21)
    assert report.net_usdt == Decimal(39)
    assert report.transaction_count == 6


def test_exact_cancellation_yields_exact_zero(conn: sqlite3.Connection) -> None:
    """A mathematically zero net from exact cancellation stays Decimal(0)."""
    persist_row(
        conn,
        transaction_date=date(2026, 9, 1),
        direction=Direction.INCOME,
        amount_usdt=Decimal("123.4500"),
        category="Работа",
        source="Проект B",
    )
    persist_row(
        conn,
        transaction_date=date(2026, 9, 2),
        direction=Direction.EXPENSE,
        amount_usdt=Decimal("123.45"),
        category="Работа",
        source="Проект B",
    )

    report = build_monthly_report(conn, year=2026, month=9)

    assert report.income_usdt == Decimal("123.4500")
    assert report.expense_usdt == Decimal("123.45")
    assert report.net_usdt == ZERO
    assert report.net_usdt == Decimal(0)
    assert not report.net_usdt.is_signed()
    project_b = source_by_name(category_by_name(report, "Работа"), "Проект B")
    assert project_b.net_usdt == ZERO
    assert project_b.transaction_count == 2
