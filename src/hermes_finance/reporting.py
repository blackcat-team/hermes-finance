"""Deterministic monthly reporting aggregation for Hermes Finance (stage E1).

This module turns the ACTIVE transactions of one exact calendar month into
an immutable, structured report of pure data: month totals, a category
breakdown, and a source breakdown inside every category. It produces no
text, no formatting, and no natural-language summary: presentation policy
belongs to a later rendering stage.

Contract highlights:

- data acquisition is delegated entirely to the accepted D1 operation
  :func:`hermes_finance.operations.list_transactions_by_month`: there is
  no competing SQL month query, no second connection, and no
  ``FinanceConfig``; the D1 visibility authority (ACTIVE-only), the D1
  year/month validation, and the D1 connection contract are inherited
  exactly, and lower-level failures (for example
  :class:`~hermes_finance.repository.RepositoryDataError`) propagate
  unchanged and are never converted into an empty report
- money is aggregated with mathematically exact arithmetic only: every
  finite :class:`decimal.Decimal` is decomposed into its exact signed
  integer coefficient and exponent, sums and differences are computed
  with arbitrary-precision Python integers, and the result is
  reconstructed exactly with the context-independent
  ``Decimal((sign, digits, exponent))`` tuple constructor. The
  aggregation therefore introduces no rounding for ANY finite Decimal
  values accepted by the domain, is completely independent of the
  ambient decimal context, and never touches binary floating point,
  ``math.fsum``, JSON numeric conversion, or quantisation
- expense amounts stay positive magnitudes and only ``net_usdt``
  applies subtraction; the net is produced by the same exact integer
  mechanism, never by ordinary contextual Decimal subtraction
- ``transaction_count`` is exactly the number of selected ACTIVE
  ``Transaction`` rows; every selected transaction contributes exactly
  once and ``processed_updates`` rows never contribute
- grouping keys are the persisted ``category``/``source`` strings exactly
  (no casefolding, no trimming, no aliasing, no classification); a
  source under two different categories remains two separate
  category-local entries and there is deliberately no global cross-
  category source merge
- breakdown ordering is deterministic and independent of insertion or
  D1 newest-first transaction order: categories are sorted by ascending
  exact category string and sources by ascending exact source string
- a valid empty month yields a zero report with ``categories == ()``,
  never ``None`` and never an exception
- reporting is entirely read-only: no rows are written, no transaction
  is started, committed, or rolled back, a caller-owned active
  transaction is left untouched, and the wall clock is never consulted
  (the caller supplies ``year``/``month`` explicitly)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Final

from hermes_finance.domain import Direction, Transaction
from hermes_finance.operations import list_transactions_by_month

__all__ = [
    "CategoryReport",
    "MonthlyFinanceReport",
    "SourceReport",
    "build_monthly_report",
]

#: Exact Decimal zero used for empty totals.
_ZERO: Final[Decimal] = Decimal(0)


@dataclass(frozen=True, slots=True)
class SourceReport:
    """Immutable per-source aggregation inside one category.

    Accumulates every selected ACTIVE transaction of one month whose
    persisted ``(category, source)`` pair is exactly this report's key.
    ``income_usdt`` and ``expense_usdt`` are exact unquantised
    ``Decimal`` sums of positive magnitudes; ``net_usdt`` is their
    exact difference; ``transaction_count`` is the number of
    contributing transactions.
    """

    source: str
    income_usdt: Decimal
    expense_usdt: Decimal
    net_usdt: Decimal
    transaction_count: int


@dataclass(frozen=True, slots=True)
class CategoryReport:
    """Immutable per-category aggregation with a source breakdown.

    The totals always equal the exact sum over :attr:`sources`, and
    ``transaction_count`` always equals the sum of the source counts:
    no transaction is counted in more than one source within its
    category.
    """

    category: str
    income_usdt: Decimal
    expense_usdt: Decimal
    net_usdt: Decimal
    transaction_count: int
    sources: tuple[SourceReport, ...]


@dataclass(frozen=True, slots=True)
class MonthlyFinanceReport:
    """Immutable month-level aggregation with a category breakdown.

    The totals always equal the exact sum over :attr:`categories`, and
    ``transaction_count`` always equals the sum of the category counts:
    every selected ACTIVE monthly transaction is represented exactly
    once. A valid month with no ACTIVE transactions yields exact zeros
    and ``categories == ()``.
    """

    year: int
    month: int
    income_usdt: Decimal
    expense_usdt: Decimal
    net_usdt: Decimal
    transaction_count: int
    categories: tuple[CategoryReport, ...]


def _decompose(value: Decimal) -> tuple[int, int]:
    """Decompose a finite Decimal into its exact ``(coefficient, exponent)``.

    The value equals ``coefficient * 10 ** exponent`` exactly, where
    ``coefficient`` is a signed arbitrary-precision Python integer.
    Non-finite values (whose Decimal tuple exponent is the string
    ``'n'``/``'N'``/``'F'`` rather than an ``int``) are rejected: they
    can never come from accepted domain amounts.
    """
    sign, digits, exponent = value.as_tuple()
    if not isinstance(exponent, int):
        raise TypeError(f"exact aggregation requires a finite Decimal, got {value!r}")
    coefficient = 0
    for digit in digits:
        coefficient = coefficient * 10 + digit
    if sign:
        coefficient = -coefficient
    return coefficient, exponent


class _ExactAccumulator:
    """Private exact accumulator for finite Decimal values.

    Holds the running total as an arbitrary-precision signed integer
    coefficient with a shared decimal exponent. Addition and
    subtraction align both operands to the smaller exponent using
    integer powers of ten and then add or subtract Python integers,
    which are exact for any magnitude. There is no rounding, no
    quantisation, no finite precision bound, and no dependence on the
    ambient decimal context: materialisation uses only the
    context-independent ``Decimal((sign, digits, exponent))`` tuple
    constructor.
    """

    __slots__ = ("_coefficient", "_exponent")

    def __init__(self) -> None:
        self._coefficient = 0
        self._exponent = 0

    def add(self, value: Decimal) -> None:
        """Add one finite Decimal exactly."""
        coefficient, exponent = _decompose(value)
        self._add_signed(coefficient, exponent)

    def subtract(self, value: Decimal) -> None:
        """Subtract one finite Decimal exactly."""
        coefficient, exponent = _decompose(value)
        self._add_signed(-coefficient, exponent)

    def _add_signed(self, coefficient: int, exponent: int) -> None:
        """Add ``coefficient * 10 ** exponent`` exactly."""
        if coefficient == 0:
            # An exact zero never changes the accumulated value.
            return
        if exponent < self._exponent:
            shift = self._exponent - exponent
            self._coefficient *= 10**shift
            self._exponent = exponent
            self._coefficient += coefficient
        elif exponent > self._exponent:
            shift = exponent - self._exponent
            self._coefficient += coefficient * 10**shift
        else:
            self._coefficient += coefficient

    def to_decimal(self) -> Decimal:
        """Materialise the exact accumulated value as a Decimal.

        Reconstruction uses the ``Decimal`` tuple constructor, which is
        exact and independent of any decimal context. An empty or fully
        cancelled accumulator yields the exact ``Decimal(0)``.
        """
        if self._coefficient == 0:
            return _ZERO
        sign = 0 if self._coefficient > 0 else 1
        digits = tuple(int(character) for character in str(abs(self._coefficient)))
        return Decimal((sign, digits, self._exponent))


@dataclass(slots=True)
class _Totals:
    """Private exact accumulation state for one grouping key.

    ``income`` and ``expense`` accumulate positive magnitudes by
    direction; ``net`` is accumulated in the same exact integer
    mechanism with income added and expense subtracted, so the net is
    never computed through ordinary contextual Decimal subtraction.
    """

    income: _ExactAccumulator = field(default_factory=_ExactAccumulator)
    expense: _ExactAccumulator = field(default_factory=_ExactAccumulator)
    net: _ExactAccumulator = field(default_factory=_ExactAccumulator)
    count: int = 0

    def add(self, transaction: Transaction) -> None:
        """Accumulate one transaction exactly once, by direction."""
        if transaction.direction is Direction.INCOME:
            self.income.add(transaction.amount_usdt)
            self.net.add(transaction.amount_usdt)
        else:
            self.expense.add(transaction.amount_usdt)
            self.net.subtract(transaction.amount_usdt)
        self.count += 1


@dataclass(slots=True)
class _MonthAccumulator:
    """Private mutable accumulation state for one month.

    Sources are grouped inside their category: the grouping key is
    effectively ``(category, source)``, so the same source string under
    different categories remains separate.
    """

    categories: dict[str, dict[str, _Totals]] = field(default_factory=dict)

    def add(self, transaction: Transaction) -> None:
        """Accumulate one transaction into its category-local source."""
        sources = self.categories.setdefault(transaction.category, {})
        totals = sources.setdefault(transaction.source, _Totals())
        totals.add(transaction)

    def build(self, year: int, month: int) -> MonthlyFinanceReport:
        """Freeze the accumulated state into an immutable report.

        Ordering is deterministic and lexical: categories by ascending
        exact category string, sources within a category by ascending
        exact source string. Insertion order and the D1 newest-first
        transaction order never influence the result.

        Category and month totals are re-aggregated from the level
        below through the same exact integer accumulators, so every
        invariant (totals equal the sum of the parts, net equals
        income minus expense) holds mathematically exactly at all
        three levels.
        """
        category_reports: list[CategoryReport] = []
        month_income = _ExactAccumulator()
        month_expense = _ExactAccumulator()
        month_net = _ExactAccumulator()
        month_count = 0
        for category in sorted(self.categories):
            sources = self.categories[category]
            source_reports: list[SourceReport] = []
            category_income = _ExactAccumulator()
            category_expense = _ExactAccumulator()
            category_net = _ExactAccumulator()
            category_count = 0
            for source in sorted(sources):
                totals = sources[source]
                source_income = totals.income.to_decimal()
                source_expense = totals.expense.to_decimal()
                source_reports.append(
                    SourceReport(
                        source=source,
                        income_usdt=source_income,
                        expense_usdt=source_expense,
                        net_usdt=totals.net.to_decimal(),
                        transaction_count=totals.count,
                    )
                )
                category_income.add(source_income)
                category_expense.add(source_expense)
                category_net.add(source_income)
                category_net.subtract(source_expense)
                category_count += totals.count
            category_reports.append(
                CategoryReport(
                    category=category,
                    income_usdt=category_income.to_decimal(),
                    expense_usdt=category_expense.to_decimal(),
                    net_usdt=category_net.to_decimal(),
                    transaction_count=category_count,
                    sources=tuple(source_reports),
                )
            )
            month_income.add(category_income.to_decimal())
            month_expense.add(category_expense.to_decimal())
            month_net.add(category_income.to_decimal())
            month_net.subtract(category_expense.to_decimal())
            month_count += category_count
        return MonthlyFinanceReport(
            year=year,
            month=month,
            income_usdt=month_income.to_decimal(),
            expense_usdt=month_expense.to_decimal(),
            net_usdt=month_net.to_decimal(),
            transaction_count=month_count,
            categories=tuple(category_reports),
        )


def build_monthly_report(
    connection: sqlite3.Connection,
    *,
    year: int,
    month: int,
) -> MonthlyFinanceReport:
    """Build the immutable monthly finance report for one exact month.

    ``connection`` must be a real caller-owned
    :class:`sqlite3.Connection` prepared by
    :func:`hermes_finance.storage.open_database`; lookalike values are
    rejected through the accepted D1 boundary. ``year``/``month``
    follow the accepted D1 validation contract exactly (real ``int``
    1..9999 and 1..12; ``bool``, ``float``, ``str``, and ``None`` are
    rejected without coercion) and ``year=9999, month=12`` is
    supported.

    The month's transactions are obtained exclusively through
    :func:`hermes_finance.operations.list_transactions_by_month`, so
    only ``TransactionStatus.ACTIVE`` rows participate: soft-deleted
    transactions contribute nothing to any total, count, or breakdown,
    and D1 remains the single visibility authority.

    Aggregation is mathematically exact for every finite ``Decimal``
    the domain accepts: values are decomposed into exact signed
    integer coefficients and exponents, combined with
    arbitrary-precision Python integers, and reconstructed through the
    context-independent ``Decimal`` tuple constructor. There is no
    quantisation, no rounding, no finite precision bound, and no
    dependence on (or mutation of) the ambient decimal context.
    ``income_usdt`` is the exact sum of the INCOME amounts,
    ``expense_usdt`` the exact sum of the EXPENSE amounts (positive
    magnitudes), and ``net_usdt`` their exact difference. A valid empty
    month returns a zero report with ``categories == ()``.

    Read-only: no rows are written, no transaction is started,
    committed, or rolled back, a caller-owned active transaction is
    left untouched (the report may observe caller-pending state exactly
    as D1 does), no pragma or migration runs, and the wall clock is
    never consulted.

    Raises
    ------
    TypeError / ValueError
        For invalid connection, year, or month values, exactly as D1
        determines them.
    RepositoryDataError
        If a selected persisted row is corrupt; the original cause is
        preserved and no partial report is returned.
    """
    transactions = list_transactions_by_month(connection, year=year, month=month)
    accumulator = _MonthAccumulator()
    for transaction in transactions:
        accumulator.add(transaction)
    return accumulator.build(year, month)
