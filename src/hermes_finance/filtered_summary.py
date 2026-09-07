"""Deterministic filtered monthly summary selection for Hermes Finance (stage G2).

This module is the smallest possible selection layer over the accepted
stage-E1 monthly report: it never aggregates, never recalculates, and
never touches the database. Given an already-built immutable
:class:`~hermes_finance.reporting.MonthlyFinanceReport`, it performs one
exact label lookup and returns the authoritative E1 report object, or
``None`` when the requested label does not exist in that month.

Contract highlights:

- selection is the only concern: the returned
  :class:`~hermes_finance.reporting.CategoryReport` /
  :class:`~hermes_finance.reporting.SourceReport` objects are the very
  immutable instances held by the E1 report. No field is copied,
  re-summed, re-derived, re-sorted, or reconstructed: every money value
  and every count reaches the caller exactly as E1 produced it, by
  object identity.
- matching is exact and case-sensitive. The only allowed normalisation
  is the accepted required-text surrounding-whitespace normalisation
  (:func:`hermes_finance.domain.normalize_required_text`) applied to the
  requested label: ``" Работа "`` matches the persisted
  ``"Работа"``, while ``"работа"``, ``"РАБОТА"``, and any other
  spelling difference do NOT match. There is deliberately no
  casefolding, lowercasing, stemming, fuzzy matching, transliteration,
  aliasing, pluralisation, or label merging, and no LLM is involved:
  natural-language interpretation belongs to the Hermes skill, never to
  the finance core.
- sources stay category-local, exactly as E1 groups them: source
  selection always happens inside one selected
  :class:`~hermes_finance.reporting.CategoryReport`, so the same source
  string under two different categories is never globally merged. There
  is deliberately no source-only selection without a category.
- a missing label is a valid result, not an error: the functions return
  ``None`` so the renderer can display an explicit "no data" state.
  No fake zero-valued report object is ever manufactured, and an exact
  zero-valued existing report is returned as the real object.
- inputs must be the real accepted types; lookalike values are rejected
  with :class:`TypeError` without duck-typing. Frozen inputs are never
  mutated: selection is a pure read-only projection.
- no SQL, no connection, no second aggregation, no transaction-list
  query, and no wall clock: the E1 report is the single authority.
"""

from __future__ import annotations

from hermes_finance.domain import normalize_required_text
from hermes_finance.reporting import (
    CategoryReport,
    MonthlyFinanceReport,
    SourceReport,
)

__all__ = [
    "select_category_source",
    "select_monthly_category",
]


def select_monthly_category(
    report: MonthlyFinanceReport,
    *,
    category: str,
) -> CategoryReport | None:
    """Select one category of a monthly report by its exact label.

    ``report`` must be a real :class:`MonthlyFinanceReport` (anything
    else is rejected with :class:`TypeError`); it is never mutated.
    ``category`` is normalised with the accepted required-text
    surrounding-whitespace normalisation (a non-string raises
    :class:`TypeError`, a blank value raises :class:`ValueError`) and is
    then compared exactly, case-sensitively, against every
    ``CategoryReport.category`` in ``report.categories``.

    Returns the authoritative immutable :class:`CategoryReport` instance
    held by the report -- never a copy, never a reconstruction -- or
    ``None`` when no category of this month carries exactly that label.
    A found report with exact zero-like totals is returned as the real
    object, deliberately distinguishable from the ``None`` of a missing
    label.
    """
    if not isinstance(report, MonthlyFinanceReport):
        raise TypeError(
            "report must be a MonthlyFinanceReport,"
            f" got {type(report).__name__!r}"
        )
    label = normalize_required_text(category, "category")
    for category_report in report.categories:
        if category_report.category == label:
            return category_report
    return None


def select_category_source(
    category_report: CategoryReport,
    *,
    source: str,
) -> SourceReport | None:
    """Select one source inside one category by its exact label.

    ``category_report`` must be a real :class:`CategoryReport` (anything
    else is rejected with :class:`TypeError`); it is never mutated.
    ``source`` is normalised with the accepted required-text
    surrounding-whitespace normalisation (a non-string raises
    :class:`TypeError`, a blank value raises :class:`ValueError`) and is
    then compared exactly, case-sensitively, against every
    ``SourceReport.source`` in ``category_report.sources``.

    The selection is deliberately category-local: only the sources of
    the supplied category participate, so the same source string under a
    different category can never contribute to or merge into this
    result. There is deliberately no source-only global selection.

    Returns the authoritative immutable :class:`SourceReport` instance
    held by the category report -- never a copy, never a reconstruction
    -- or ``None`` when no source of this category carries exactly that
    label.
    """
    if not isinstance(category_report, CategoryReport):
        raise TypeError(
            "category_report must be a CategoryReport,"
            f" got {type(category_report).__name__!r}"
        )
    label = normalize_required_text(source, "source")
    for source_report in category_report.sources:
        if source_report.source == label:
            return source_report
    return None
