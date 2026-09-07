"""High-level integration facade for Hermes Finance (stages F1, F2, G1, G2).

This module composes the already accepted lower layers into small
deterministic operations that the later Hermes application can call:

- stage F1 intake: :func:`ingest_finance_message`
- stage F2 monthly report: :func:`get_monthly_report_text`
- stage G1 transaction lists: :func:`get_recent_transactions_text`,
  :func:`get_transactions_by_date_text`, and
  :func:`get_transactions_by_month_text`
- stage G2 filtered monthly summaries:
  :func:`get_monthly_category_summary_text` and
  :func:`get_monthly_source_summary_text`

The F1 composition is:

- stage B1: :func:`hermes_finance.parser.parse_transaction_input`
- stage B2: :func:`hermes_finance.ledger.create_transaction`
- stage C3: :func:`hermes_finance.ingest.ingest_transaction`

The F2 composition is:

- stage E1: :func:`hermes_finance.reporting.build_monthly_report`
- stage E2: :func:`hermes_finance.rendering.render_monthly_report`

Each G1 composition is:

- stage D1: the accepted
  :mod:`hermes_finance.operations` list operation
- stage G1: the matching accepted
  :mod:`hermes_finance.rendering` transaction-list renderer

Each G2 composition is:

- stage E1: :func:`hermes_finance.reporting.build_monthly_report`
- stage G2: the accepted
  :mod:`hermes_finance.filtered_summary` exact selection plus the
  matching accepted :mod:`hermes_finance.rendering` summary renderer

The first seam is:

    Telegram message text
    TelegramMessageRef provenance
    business date
    received timestamp
        -> :func:`ingest_finance_message`
        -> :class:`hermes_finance.ingest.IngestResult`

The second seam is:

    exact calendar year and month
        -> :func:`get_monthly_report_text`
        -> plain-text monthly report :class:`str`

The G1 seams are:

    optional limit / exact business date / exact calendar month
        -> :func:`get_recent_transactions_text`
        -> :func:`get_transactions_by_date_text`
        -> :func:`get_transactions_by_month_text`
        -> plain-text transaction list :class:`str`

The G2 seams are:

    exact calendar year and month plus one exact category label
    (optionally plus one exact category-local source label)
        -> :func:`get_monthly_category_summary_text`
        -> :func:`get_monthly_source_summary_text`
        -> plain-text filtered summary card :class:`str`

All operations own orchestration only. Every business rule -- grammar
parsing, amount normalisation, transaction construction, idempotency
classification, duplicate handling, persistence, month selection,
visibility, aggregation, report rendering, list selection, list
rendering, exact label selection, and summary rendering -- stays in its
accepted authoritative lower layer and is never duplicated here.

Boundaries:

- no wall clock: ``transaction_date``, ``received_at``, ``year``,
  ``month``, ``limit``, and every category/source label are always
  supplied explicitly by the caller and never derived from any clock or
  calendar
- no persistence of its own: the caller-owned
  :class:`sqlite3.Connection` is handed to the accepted services
  unchanged, together with its connection, ownership, and migration
  contract
- no Telegram library dependency: only the domain-level
  :class:`hermes_finance.provenance.TelegramMessageRef` is accepted;
  Hermes owns construction of the provenance reference from its own
  Telegram update, and the returned texts are plain ``str`` that
  Hermes may transport later
- no user-facing output of its own: parser errors, consistency errors,
  and the ingest dispositions of the accepted layers propagate
  unchanged, and every text result is exactly the accepted renderer's
  output with nothing prepended or appended
- no mutation of any caller input
- no aggregation, selection, or formatting of its own: the F2 report
  operation never queries transactions, sums or differences values,
  counts rows, or sorts categories or sources, the G1 list operations
  never add SQL, filtering, re-sorting, or reaggregation on top of the
  accepted D1 selection, and the G2 summary operations never add a
  second aggregation, a transaction-list query, or any label matching
  beyond the accepted E1/G2 layers; the accepted D1/E1/E2/G2 layers
  remain the single authorities
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime

from hermes_finance.filtered_summary import (
    select_category_source,
    select_monthly_category,
)
from hermes_finance.ingest import IngestResult, ingest_transaction
from hermes_finance.ledger import create_transaction
from hermes_finance.operations import (
    DEFAULT_LIMIT,
    list_recent_transactions,
    list_transactions_by_date,
    list_transactions_by_month,
)
from hermes_finance.parser import parse_transaction_input
from hermes_finance.provenance import TelegramMessageRef
from hermes_finance.rendering import (
    render_monthly_category_summary,
    render_monthly_report,
    render_monthly_source_summary,
    render_recent_transactions,
    render_transactions_by_date,
    render_transactions_by_month,
)
from hermes_finance.reporting import build_monthly_report

__all__ = [
    "get_monthly_category_summary_text",
    "get_monthly_report_text",
    "get_monthly_source_summary_text",
    "get_recent_transactions_text",
    "get_transactions_by_date_text",
    "get_transactions_by_month_text",
    "ingest_finance_message",
]


def ingest_finance_message(
    connection: sqlite3.Connection,
    text: str,
    provenance: TelegramMessageRef,
    *,
    transaction_date: date,
    received_at: datetime,
) -> IngestResult:
    """Parse, create, and idempotently ingest one Finance topic message.

    This is the single intake seam for the future Hermes Telegram
    ingress: the caller supplies the raw message text, the already
    constructed Telegram provenance reference, the business date, and
    the receive timestamp; the accepted parser, creation core, and
    idempotent ingest service do all the work.

    Contract:

    - ``connection``: a caller-owned :class:`sqlite3.Connection`
      prepared by :func:`hermes_finance.storage.open_database`; it is
      passed through to the accepted ingest service unchanged, and the
      lower-layer connection contract (including refusal while the
      caller holds an active transaction) remains authoritative
    - ``text``: the raw message text, handed to the accepted parser
      exactly as provided; parser semantics (grammar, currency rules,
      text normalisation) and :class:`TransactionParseError` failures
      propagate unchanged
    - ``provenance``: a :class:`TelegramMessageRef`; it is passed
      through to the accepted ingest service unchanged
    - ``transaction_date``: the business date, supplied explicitly by
      the caller (Hermes derives it from its routing/runtime context);
      it is used exactly as given and never derived from
      ``received_at`` or any clock
    - ``received_at``: the timezone-aware receive timestamp; the same
      value becomes the transaction ``created_at`` (and therefore the
      initial ``updated_at``) and the processed-update ``processed_at``

    Returns the :class:`~hermes_finance.ingest.IngestResult` of the
    accepted ingest service unchanged: ``CREATED``,
    ``DUPLICATE_MESSAGE``, and ``DUPLICATE_UPDATE`` dispositions, the
    persisted transaction, and :class:`IngestConsistencyError` /
    unproven :class:`sqlite3.IntegrityError` failure semantics are all
    preserved exactly. On duplicate logical-message delivery the
    persisted transaction wins: this function never compares, mutates,
    or reconciles business fields of an already persisted transaction.

    Raises whatever the accepted lower layers raise, unchanged,
    including :class:`hermes_finance.parser.TransactionParseError` for
    malformed text, domain ``TypeError``/``ValueError`` for invalid
    dates or naive timestamps, and the ingest service's failure types.
    """

    parsed = parse_transaction_input(text)

    transaction = create_transaction(
        parsed,
        transaction_date=transaction_date,
        created_at=received_at,
    )

    return ingest_transaction(
        connection,
        transaction,
        provenance,
        processed_at=received_at,
    )


def get_monthly_report_text(
    connection: sqlite3.Connection,
    *,
    year: int,
    month: int,
) -> str:
    """Return the deterministic plain-text monthly finance report.

    This is the single monthly-report seam for the later Hermes
    application: the caller supplies a real caller-owned
    :class:`sqlite3.Connection` and the exact calendar ``year`` and
    ``month``; the accepted E1 aggregation
    (:func:`hermes_finance.reporting.build_monthly_report`) builds the
    report and the accepted E2 renderer
    (:func:`hermes_finance.rendering.render_monthly_report`) turns it
    into text.

    Contract:

    - ``connection``: a caller-owned :class:`sqlite3.Connection`
      prepared by :func:`hermes_finance.storage.open_database`; it is
      handed to the accepted report builder unchanged, and the
      lower-layer read-only connection contract remains authoritative
    - ``year`` / ``month``: the exact calendar month; both values are
      passed through unchanged and are never clamped, normalised,
      coerced, or derived from any clock
    - the return value is the exact ``str`` produced by
      :func:`hermes_finance.rendering.render_monthly_report`, with
      nothing prepended, appended, wrapped, or reformatted

    Raises whatever the accepted lower layers raise, unchanged,
    including the E1/D1 year and month validation failures
    (:class:`TypeError` / :class:`ValueError`) and
    :class:`hermes_finance.repository.RepositoryDataError` for corrupt
    persisted rows.
    """

    report = build_monthly_report(
        connection,
        year=year,
        month=month,
    )
    return render_monthly_report(report)


def get_recent_transactions_text(
    connection: sqlite3.Connection,
    *,
    limit: int = DEFAULT_LIMIT,
) -> str:
    """Return the deterministic plain-text "last N transactions" view.

    This is the recent-list seam for the later Hermes application: the
    caller supplies a real caller-owned
    :class:`sqlite3.Connection` and the explicit ``limit``; the
    accepted D1 operation
    :func:`hermes_finance.operations.list_recent_transactions`
    selects the newest ACTIVE transactions and the accepted G1
    renderer :func:`hermes_finance.rendering.render_recent_transactions`
    turns them into text.

    Contract:

    - ``connection``: a caller-owned :class:`sqlite3.Connection`
      prepared by :func:`hermes_finance.storage.open_database`; it is
      handed to the accepted D1 operation unchanged, and the
      lower-layer read-only connection contract remains authoritative
    - ``limit``: defaults to the accepted D1 default (10) and is
      passed through unchanged; the accepted D1 limit validation
      (real ``int`` between 1 and 100) is authoritative and its
      failures propagate unchanged
    - the return value is the exact ``str`` produced by the accepted
      renderer, with nothing prepended, appended, wrapped, or
      reformatted

    Raises whatever the accepted lower layers raise, unchanged,
    including D1 limit validation failures and
    :class:`hermes_finance.repository.RepositoryDataError` for corrupt
    persisted rows.
    """

    transactions = list_recent_transactions(connection, limit=limit)
    return render_recent_transactions(transactions)


def get_transactions_by_date_text(
    connection: sqlite3.Connection,
    transaction_date: date,
) -> str:
    """Return the deterministic plain-text view of one exact date.

    This is the exact-date list seam for the later Hermes application:
    the caller supplies a real caller-owned
    :class:`sqlite3.Connection` and the exact business
    ``transaction_date``; the accepted D1 operation
    :func:`hermes_finance.operations.list_transactions_by_date`
    selects the ACTIVE transactions of that one date and the accepted
    G1 renderer
    :func:`hermes_finance.rendering.render_transactions_by_date` turns
    them into text.

    Contract:

    - ``connection``: a caller-owned :class:`sqlite3.Connection`
      prepared by :func:`hermes_finance.storage.open_database`; it is
      handed to the accepted D1 operation unchanged, and the
      lower-layer read-only connection contract remains authoritative
    - ``transaction_date``: the exact business date, supplied
      explicitly by the caller; it is used exactly as given and never
      derived from any clock
    - the return value is the exact ``str`` produced by the accepted
      renderer, with nothing prepended, appended, wrapped, or
      reformatted

    Raises whatever the accepted lower layers raise, unchanged,
    including the D1/domain date validation failures and
    :class:`hermes_finance.repository.RepositoryDataError` for corrupt
    persisted rows.
    """

    transactions = list_transactions_by_date(connection, transaction_date)
    return render_transactions_by_date(transactions, transaction_date)


def get_transactions_by_month_text(
    connection: sqlite3.Connection,
    *,
    year: int,
    month: int,
) -> str:
    """Return the deterministic plain-text view of one exact month.

    This is the exact-month list seam for the later Hermes application:
    the caller supplies a real caller-owned
    :class:`sqlite3.Connection` and the exact calendar ``year`` and
    ``month``; the accepted D1 operation
    :func:`hermes_finance.operations.list_transactions_by_month`
    selects the ACTIVE transactions of that one calendar month and the
    accepted G1 renderer
    :func:`hermes_finance.rendering.render_transactions_by_month`
    turns them into text.

    Contract:

    - ``connection``: a caller-owned :class:`sqlite3.Connection`
      prepared by :func:`hermes_finance.storage.open_database`; it is
      handed to the accepted D1 operation unchanged, and the
      lower-layer read-only connection contract remains authoritative
    - ``year`` / ``month``: the exact calendar month; both values are
      passed through unchanged and are never clamped, normalised,
      coerced, or derived from any clock
    - the return value is the exact ``str`` produced by the accepted
      renderer, with nothing prepended, appended, wrapped, or
      reformatted

    Raises whatever the accepted lower layers raise, unchanged,
    including the D1 year and month validation failures and
    :class:`hermes_finance.repository.RepositoryDataError` for corrupt
    persisted rows.
    """

    transactions = list_transactions_by_month(connection, year=year, month=month)
    return render_transactions_by_month(transactions, year=year, month=month)


def get_monthly_category_summary_text(
    connection: sqlite3.Connection,
    *,
    year: int,
    month: int,
    category: str,
) -> str:
    """Return the deterministic plain-text summary of one category.

    This is the filtered category-summary seam for the later Hermes
    application: the caller supplies a real caller-owned
    :class:`sqlite3.Connection`, the exact calendar ``year`` and
    ``month``, and the exact ``category`` label; the accepted E1
    aggregation (:func:`hermes_finance.reporting.build_monthly_report`)
    builds the single authoritative monthly report, the accepted G2
    selection
    (:func:`hermes_finance.filtered_summary.select_monthly_category`)
    picks the category-local report object by exact label, and the
    accepted G2 renderer
    (:func:`hermes_finance.rendering.render_monthly_category_summary`)
    turns it into text. A category that does not exist in the selected
    month renders the explicit ``no data`` state; no zero report is
    manufactured.

    Contract:

    - ``connection``: a caller-owned :class:`sqlite3.Connection`
      prepared by :func:`hermes_finance.storage.open_database`; it is
      handed to the accepted report builder unchanged, and the
      lower-layer read-only connection contract remains authoritative
    - ``year`` / ``month`` / ``category``: used exactly as given and
      never clamped, coerced, case-folded, or derived from any clock;
      the accepted E1 year/month validation and the accepted G2 label
      normalisation and exact matching stay authoritative
    - no second aggregation, no transaction-list query, and no SQL: the
      E1 report is the single money authority and its values reach the
      text verbatim
    - the return value is the exact ``str`` produced by the accepted
      renderer, with nothing prepended, appended, wrapped, or
      reformatted

    Raises whatever the accepted lower layers raise, unchanged,
    including the E1/D1 year and month validation failures
    (:class:`TypeError` / :class:`ValueError`), the G2 label validation
    failures (:class:`TypeError` for a non-string, :class:`ValueError`
    for a blank label), and
    :class:`hermes_finance.repository.RepositoryDataError` for corrupt
    persisted rows.
    """

    report = build_monthly_report(
        connection,
        year=year,
        month=month,
    )
    category_report = select_monthly_category(report, category=category)
    return render_monthly_category_summary(
        year=year,
        month=month,
        category=category,
        category_report=category_report,
    )


def get_monthly_source_summary_text(
    connection: sqlite3.Connection,
    *,
    year: int,
    month: int,
    category: str,
    source: str,
) -> str:
    """Return the deterministic plain-text summary of one category source.

    This is the filtered category-local source-summary seam for the
    later Hermes application: the caller supplies a real caller-owned
    :class:`sqlite3.Connection`, the exact calendar ``year`` and
    ``month``, the exact ``category`` label, and the exact ``source``
    label inside that category. The accepted E1 aggregation
    (:func:`hermes_finance.reporting.build_monthly_report`) builds the
    single authoritative monthly report, the accepted G2 selection
    (:func:`hermes_finance.filtered_summary.select_monthly_category`
    and
    :func:`hermes_finance.filtered_summary.select_category_source`)
    picks the category-local source report object by exact labels, and
    the accepted G2 renderer
    (:func:`hermes_finance.rendering.render_monthly_source_summary`)
    turns it into text. A missing category or a missing source renders
    the explicit ``no data`` state; no zero report is manufactured, and
    sources of other categories never contribute.

    ``source`` always means a source inside ``category``: there is
    deliberately no source-only global summary, matching the accepted
    E1 category-local grouping semantics.

    Contract:

    - ``connection``: a caller-owned :class:`sqlite3.Connection`
      prepared by :func:`hermes_finance.storage.open_database`; it is
      handed to the accepted report builder unchanged, and the
      lower-layer read-only connection contract remains authoritative
    - ``year`` / ``month`` / ``category`` / ``source``: used exactly as
      given and never clamped, coerced, case-folded, or derived from
      any clock; the accepted E1 year/month validation and the accepted
      G2 label normalisation and exact matching stay authoritative
    - no second aggregation, no transaction-list query, and no SQL: the
      E1 report is the single money authority and its values reach the
      text verbatim
    - the return value is the exact ``str`` produced by the accepted
      renderer, with nothing prepended, appended, wrapped, or
      reformatted

    Raises whatever the accepted lower layers raise, unchanged,
    including the E1/D1 year and month validation failures
    (:class:`TypeError` / :class:`ValueError`), the G2 label validation
    failures (:class:`TypeError` for a non-string, :class:`ValueError`
    for a blank label), and
    :class:`hermes_finance.repository.RepositoryDataError` for corrupt
    persisted rows.
    """

    report = build_monthly_report(
        connection,
        year=year,
        month=month,
    )
    category_report = select_monthly_category(report, category=category)
    source_report = (
        select_category_source(category_report, source=source)
        if category_report is not None
        else None
    )
    return render_monthly_source_summary(
        year=year,
        month=month,
        category=category,
        source=source,
        source_report=source_report,
    )
