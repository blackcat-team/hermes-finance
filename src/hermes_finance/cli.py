"""Finance-owned CLI boundary for Hermes Finance (stages F3, G1, G2, G3, H1).

This module is the ``hermes-finance`` console script: a thin subprocess
boundary that lets an external caller (for example a Hermes skill or a
tiny adapter) reach the accepted finance core without any in-process
coupling to the Hermes host:

    Hermes / skill / tiny adapter
            | subprocess
            v
       hermes-finance CLI
             |
             v
        accepted finance core (F1 / F2 / G1 / G2 / D2 / H1)
             |
             v
        SQLite

Five command families exist:

- ``hermes-finance ingest-telegram ...``: a thin boundary over the
  accepted stage-F1 intake
  :func:`hermes_finance.integration.ingest_finance_message`
- ``hermes-finance month --year Y --month M`` (or
  ``hermes-finance month --relative current|previous --timezone ZONE``):
  a thin boundary over the accepted stage-F2 aggregated monthly report
  :func:`hermes_finance.integration.get_monthly_report_text`
- ``hermes-finance transactions <subcommand>``: a thin boundary over
  the accepted stage-G1 individual-transaction list facades:
  ``transactions recent [--limit N]`` for the newest transactions,
  ``transactions date --date YYYY-MM-DD`` (or
  ``transactions date --relative today|yesterday --timezone ZONE``) for
  one exact business date, and ``transactions month --year Y --month M``
  (or ``transactions month --relative current|previous --timezone
  ZONE``) for the individual transactions of one exact calendar month
- ``hermes-finance summary month --year Y --month M --category CATEGORY
  [--source SOURCE]`` (or the ``--relative current|previous
  --timezone ZONE`` alternative): a thin boundary over the accepted
  stage-G2 filtered monthly summary facades
  :func:`hermes_finance.integration.get_monthly_category_summary_text`
  and
  :func:`hermes_finance.integration.get_monthly_source_summary_text`
- ``hermes-finance transaction <subcommand>``: a thin boundary over the
  accepted stage-D2 mutation core (stage H1):
  ``transaction edit-amount (--id ID | --last) --amount AMOUNT`` for
  the amount-only correction of one persisted ACTIVE transaction
  through :func:`hermes_finance.corrections.edit_transaction_amount`,
  and ``transaction delete (--id ID | --last)`` for the idempotent
  soft deletion of one transaction through
  :func:`hermes_finance.mutations.soft_delete_transaction`

``month`` and ``transactions month`` stay deliberately distinct: the
former prints the aggregated monthly report, the latter prints the
individual operations of that month. ``summary month`` prints the
aggregated totals of one exact category (optionally one exact
category-local source) of one exact month, selected from the same
accepted E1 aggregation; it never recalculates money itself. The
singular ``transaction`` family is the only mutation surface: every
write goes through the accepted D2 mutation core, there is no direct
SQL, and there is deliberately no hard delete.

Relative periods (stage G3) extend the same commands instead of
creating competing query paths: the CLI maps the canonical relative
token to the exact period and then reuses the identical accepted
facades, so rendering, SQL, and report math never change.

Boundaries:

- stdlib only: :mod:`argparse` for the command line, :mod:`json` for
  the machine-readable ingest and mutation output; no external CLI
  framework, no network, no service, no scheduler
- update-neutral: the module never imports or references the Hermes
  host application or any Hermes installation, and it has no
  third-party Telegram client dependency; it runs entirely from the
  Hermes-finance environment
- the database path is read from the ``HERMES_FINANCE_DB_PATH``
  environment variable at invocation time; a missing, empty, or
  whitespace-only value is a deterministic error and no default path is
  invented
- the database is always opened through the accepted bootstrap
  (:func:`hermes_finance.storage.open_database` with a
  :class:`~hermes_finance.config.FinanceConfig`) and the connection is
  always closed, on both success and failure paths
- explicit periods need no clock: the business date, the receive
  timestamp, the report year/month, the list year/month, and the
  summary year/month are explicit caller inputs and are never derived
  from any clock or calendar
- relative periods capture exactly one clock snapshot: a relative
  invocation reads the clock once through the dedicated
  :func:`_capture_reference_time` seam (a single aware UTC instant)
  and resolves every period from that snapshot through the pure
  :mod:`hermes_finance.periods` resolver; the clock is never read more
  than once per invocation, the server-local timezone is never
  consulted, and the model never performs date arithmetic
- mutations capture exactly one clock snapshot through the same
  :func:`_capture_reference_time` seam: one aware UTC instant per
  mutation invocation becomes the edit ``updated_at`` or the delete
  ``deleted_at``; the D2 mutation core keeps requiring an explicit
  timezone-aware timestamp, the server-local timezone is never
  consulted, and no business timezone is required for a mutation
- ``--relative`` and the explicit period arguments are mutually
  exclusive alternatives; supplying both, supplying neither, or
  supplying only one of ``--year`` / ``--month`` is a deterministic
  :class:`FinanceCliError` before any database work happens
- mutation targets are mutually exclusive alternatives: ``--id`` and
  ``--last``; supplying both or supplying neither is a deterministic
  :class:`FinanceCliError` before any database work happens
- ``--id`` is a transport string handed to the accepted layers
  unchanged: the accepted repository ID validation is authoritative
  and no second ID parser exists here
- ``--last`` resolves through the accepted D1 operation
  :func:`hermes_finance.operations.list_recent_transactions` with
  ``limit=1`` (exactly the ``transactions recent --limit 1``
  semantics); there is deliberately no competing "latest" SQL, and an
  empty ACTIVE ledger is a deterministic :class:`FinanceCliError`
  before any mutation happens
- ``--timezone`` is required exactly when ``--relative`` is used, must
  be a real IANA timezone name parsed through stdlib
  ``zoneinfo.ZoneInfo``, has no default, and no UTC fallback exists; an
  unknown or unavailable zone is a deterministic
  :class:`FinanceCliError` raised before any database query runs; the
  mutation commands never need a ``--timezone`` value
- transport parsing only: the CLI parses integer CLI identifiers, one
  ISO ``YYYY-MM-DD`` date, one ISO-8601 timezone-aware datetime, and
  one IANA timezone name; it never parses or reimplements finance
  grammar (``+``/``-``, amounts, categories, sources, comments) or
  financial totals, which stay in the accepted core; ``--amount`` is
  the positive magnitude handed to the accepted domain amount
  validation unchanged (the stored income/expense direction is
  preserved and never silently flipped); a naive ``--received-at`` is
  rejected
- category and source labels are transport strings: ``--category`` is
  required and ``--source`` is optional, and both are handed to the
  accepted G2 facades exactly as supplied; the CLI never normalises,
  case-folds, aliases, or matches labels itself and never calculates
  money from the returned metrics
- mutation output is one deterministic JSON object on stdout (the
  disposition and the transaction ID only, exactly analogous to the
  ingest boundary); no amount recomputation happens here and the JSON
  is printed only after the accepted mutation core succeeded, so a
  failed mutation never produces a success JSON
- real Telegram thread provenance: ``--thread-id`` is required with no
  default, no fallback, and no inferred value; the calling boundary
  must supply the actual Telegram ``message_thread_id`` so no
  fabricated provenance value can ever reach the accepted core
- accepted-layer failures (for example
  :class:`hermes_finance.parser.TransactionParseError` for invalid
  finance text, or the accepted D2 mutation errors for a missing,
  malformed, or deleted target) propagate unchanged and cause no write
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import zoneinfo
from collections.abc import Sequence
from datetime import UTC, date, datetime, tzinfo
from pathlib import Path
from typing import Final

from hermes_finance.config import FinanceConfig
from hermes_finance.corrections import edit_transaction_amount
from hermes_finance.integration import (
    get_monthly_category_summary_text,
    get_monthly_report_text,
    get_monthly_source_summary_text,
    get_recent_transactions_text,
    get_transactions_by_date_text,
    get_transactions_by_month_text,
    ingest_finance_message,
)
from hermes_finance.mutations import soft_delete_transaction
from hermes_finance.operations import DEFAULT_LIMIT, list_recent_transactions
from hermes_finance.periods import (
    DATE_PERIODS,
    MONTH_PERIODS,
    resolve_relative_date,
    resolve_relative_month,
)
from hermes_finance.provenance import TelegramMessageRef
from hermes_finance.storage import open_database

__all__ = [
    "HERMES_FINANCE_DB_PATH",
    "FinanceCliError",
    "main",
]


#: Environment variable holding the finance database file path. It is
#: read at invocation time, never at import time.
HERMES_FINANCE_DB_PATH: Final[str] = "HERMES_FINANCE_DB_PATH"

#: Deterministic business timezone required by the accepted config
#: value object. The ingest, monthly-report, transaction-list, and
#: filtered-summary paths never consult it: their dates, timestamps,
#: and months are explicit caller inputs, and relative periods resolve
#: through the separately supplied ``--timezone`` IANA zone instead.
_CLI_BUSINESS_TIMEZONE: Final = UTC

#: Exact ISO ``YYYY-MM-DD`` transport representation accepted for
#: ``--transaction-date``.
_ISO_DATE_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


class FinanceCliError(Exception):
    """Deterministic CLI boundary failure.

    Covers the failures this CLI owns: a missing/blank/whitespace-only
    ``HERMES_FINANCE_DB_PATH`` value, an unparsable transport
    representation (``--transaction-date`` / ``--received-at``), a naive
    ``--received-at`` timestamp, an invalid relative-period argument
    combination (explicit period and ``--relative`` together, neither of
    them, or only one of ``--year`` / ``--month``), a relative
    invocation without ``--timezone``, an unknown or unavailable
    IANA ``--timezone`` name, an invalid mutation target combination
    (``--id`` and ``--last`` together or neither of them), and a
    ``--last`` mutation over a ledger with no ACTIVE transaction.
    ``main`` reports the message on stderr and exits with status 1;
    accepted finance-layer failures are deliberately not wrapped and
    propagate unchanged.
    """


def _resolve_database_path() -> Path:
    """Read the finance database path from the environment, now.

    Missing, empty, and whitespace-only values are rejected with
    :class:`FinanceCliError`. The value is otherwise used exactly as
    supplied; no default or server path is invented.
    """
    raw_path = os.environ.get(HERMES_FINANCE_DB_PATH)
    if raw_path is None or not raw_path.strip():
        raise FinanceCliError(
            f"the {HERMES_FINANCE_DB_PATH} environment variable must be"
            " set to the finance database file path"
        )
    return Path(raw_path.strip())


def _parse_iso_date(value: str, option_name: str) -> date:
    """Parse one ISO ``YYYY-MM-DD`` CLI date value.

    Anything that is not exactly ten characters ``YYYY-MM-DD`` digits
    and dashes, or that is not a real calendar date, is rejected with
    :class:`FinanceCliError` naming ``option_name``.
    """
    if _ISO_DATE_PATTERN.fullmatch(value) is None:
        raise FinanceCliError(
            f"{option_name} must be an ISO YYYY-MM-DD date,"
            f" got {value!r}"
        )
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise FinanceCliError(
            f"{option_name} is not a real calendar date: {value!r}"
        ) from error


def _parse_transaction_date(value: str) -> date:
    """Parse ``--transaction-date`` from its ISO ``YYYY-MM-DD`` form."""
    return _parse_iso_date(value, "--transaction-date")


def _parse_received_at(value: str) -> datetime:
    """Parse ``--received-at`` from its ISO-8601 representation.

    The value must parse as an ISO-8601 datetime and must carry an
    explicit UTC offset: a naive timestamp is rejected with
    :class:`FinanceCliError` before any database work happens.
    """
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise FinanceCliError(
            f"--received-at must be an ISO-8601 datetime, got {value!r}"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise FinanceCliError(
            f"--received-at must be timezone-aware, got {value!r}"
        )
    return parsed


def _capture_reference_time() -> datetime:
    """Capture the single current-time snapshot for this invocation.

    This is the only wall-clock access in the whole CLI. A relative
    invocation calls it exactly once and resolves every period from
    that one aware UTC instant through the pure
    :mod:`hermes_finance.periods` resolver. A mutation invocation
    (stage H1) calls it exactly once too: the single aware UTC snapshot
    becomes the edit ``updated_at`` or the delete ``deleted_at``
    timestamp of the accepted D2 mutation core. Explicit-period and
    ingest commands never call it. The server-local timezone is never
    consulted: the snapshot is taken in UTC and converted only through
    the explicitly supplied business timezone during period resolution,
    and never converted at all for mutations.
    """
    return datetime.now(UTC)


def _parse_business_timezone(value: str | None) -> tzinfo:
    """Parse ``--timezone`` into a stdlib ``zoneinfo.ZoneInfo``.

    A missing, blank, or whitespace-only value is rejected: a relative
    period is undefined without a business timezone and no default or
    UTC fallback is ever assumed. An unknown or unavailable IANA name is
    rejected deterministically through the real stdlib
    ``zoneinfo.ZoneInfo`` lookup before any database work happens;
    another timezone is never guessed.
    """
    if value is None or not value.strip():
        raise FinanceCliError(
            "--timezone is required with --relative and must be an IANA"
            " timezone name; no default timezone is assumed"
        )
    name = value.strip()
    try:
        return zoneinfo.ZoneInfo(name)
    except (zoneinfo.ZoneInfoNotFoundError, ValueError) as error:
        raise FinanceCliError(
            f"--timezone is not a valid IANA timezone name: {name!r}"
        ) from error


def _resolve_transactions_date(args: argparse.Namespace) -> date:
    """Resolve the ``transactions date`` period: explicit or relative.

    ``--date`` and ``--relative`` are mutually exclusive alternatives;
    exactly one of them must be supplied. The explicit ISO date is
    parsed strictly as before. The relative token resolves the single
    captured clock snapshot through the pure period resolver in the
    required business timezone and returns the exact business date,
    which then flows into the identical accepted G1 facade.
    """
    relative: str | None = args.relative
    if relative is not None:
        if args.date is not None:
            raise FinanceCliError(
                "transactions date: --date and --relative are mutually"
                " exclusive; supply either one, not both"
            )
        business_timezone = _parse_business_timezone(args.timezone)
        reference_time = _capture_reference_time()
        return resolve_relative_date(
            relative,
            reference_time=reference_time,
            business_timezone=business_timezone,
        )
    if args.date is None:
        raise FinanceCliError(
            "transactions date: either --date or --relative is required"
        )
    return _parse_iso_date(args.date, "--date")


def _resolve_month_period(
    args: argparse.Namespace, command_hint: str
) -> tuple[int, int]:
    """Resolve a month command's period: explicit pair or relative token.

    ``--year`` and ``--month`` behave as one pair and are mutually
    exclusive with ``--relative``; exactly one alternative must be
    supplied. The relative token resolves the single captured clock
    snapshot through the pure period resolver in the required business
    timezone and returns the exact ``(year, month)``, which then flows
    into the identical accepted facade of the calling command.
    """
    year: int | None = args.year
    month: int | None = args.month
    relative: str | None = args.relative
    if relative is not None:
        if year is not None or month is not None:
            raise FinanceCliError(
                f"{command_hint}: --relative and --year/--month are"
                " mutually exclusive; supply either the explicit pair or"
                " the relative token, not both"
            )
        business_timezone = _parse_business_timezone(args.timezone)
        reference_time = _capture_reference_time()
        return resolve_relative_month(
            relative,
            reference_time=reference_time,
            business_timezone=business_timezone,
        )
    if year is None and month is None:
        raise FinanceCliError(
            f"{command_hint}: either --year and --month together or"
            " --relative is required"
        )
    if year is None or month is None:
        raise FinanceCliError(
            f"{command_hint}: --year and --month must be supplied together"
        )
    return year, month


def _open_finance_database(database_path: Path) -> sqlite3.Connection:
    """Open the finance database through the accepted bootstrap."""
    return open_database(
        FinanceConfig(
            database_path=database_path,
            business_timezone=_CLI_BUSINESS_TIMEZONE,
        )
    )


def _require_exactly_one_target(
    args: argparse.Namespace, command_hint: str
) -> None:
    """Validate the mutation target combination before any database work.

    ``--id`` and ``--last`` are mutually exclusive alternatives; exactly
    one of them must be supplied. Supplying both or supplying neither
    is a deterministic :class:`FinanceCliError` raised before the
    database is opened, so no mutation can ever happen for an invalid
    target combination.
    """
    if args.id is not None and args.last:
        raise FinanceCliError(
            f"{command_hint}: --id and --last are mutually exclusive;"
            " supply either one, not both"
        )
    if args.id is None and not args.last:
        raise FinanceCliError(
            f"{command_hint}: either --id or --last is required"
        )


def _resolve_transaction_target(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
    command_hint: str,
) -> str:
    """Resolve the mutation target ID: the exact ``--id`` or ``--last``.

    The exact ``--id`` value is handed through unchanged as a transport
    string: the accepted repository ID validation stays authoritative
    and there is no second ID parser here. ``--last`` resolves through
    the accepted D1 operation with ``limit=1`` -- exactly the
    ``transactions recent --limit 1`` semantics, with no competing
    "latest" SQL -- and fails deterministically before any mutation
    when the ledger has no ACTIVE transaction.
    """
    if args.id is not None:
        exact_id: object = args.id
        if not isinstance(exact_id, str):
            raise FinanceCliError(
                f"{command_hint}: --id must be a transaction ID string"
            )
        return exact_id
    recent = list_recent_transactions(connection, limit=1)
    if not recent:
        raise FinanceCliError(
            f"{command_hint}: --last requires at least one ACTIVE"
            " transaction; the ledger has none"
        )
    target_id = recent[0].transaction_id
    if target_id is None:
        # A persisted row always carries its repository ID; this guard
        # keeps the failure deterministic instead of silently mutating
        # an unpersisted lookalike.
        raise FinanceCliError(
            f"{command_hint}: --last resolved to a transaction without a"
            " persisted transaction_id"
        )
    return target_id


def _run_transaction_edit_amount(args: argparse.Namespace) -> int:
    """Execute ``transaction edit-amount``: H1 delegation, JSON on stdout.

    The target is either the exact ``--id`` or the latest ACTIVE
    transaction (``--last``, resolved through the accepted D1 recent
    operation). ``--amount`` is the positive magnitude handed to the
    accepted domain amount validation unchanged: the stored
    income/expense direction is preserved and never silently flipped.
    One aware UTC clock snapshot becomes the mutation ``updated_at``.
    On success exactly one JSON object with the ``UPDATED`` disposition
    and the transaction ID is printed; on any failure nothing is
    printed on stdout and no success is claimed.
    """
    command_hint = "transaction edit-amount"
    database_path = _resolve_database_path()
    _require_exactly_one_target(args, command_hint)

    connection = _open_finance_database(database_path)
    try:
        transaction_id = _resolve_transaction_target(connection, args, command_hint)
        mutation_time = _capture_reference_time()
        edited = edit_transaction_amount(
            connection,
            transaction_id,
            amount_usdt=args.amount,
            updated_at=mutation_time,
        )
    finally:
        connection.close()

    outcome = {
        "disposition": "UPDATED",
        "transaction_id": edited.transaction_id,
    }
    print(json.dumps(outcome))
    return 0


def _run_transaction_delete(args: argparse.Namespace) -> int:
    """Execute ``transaction delete``: D2 soft delete, JSON on stdout.

    The target is either the exact ``--id`` or the latest ACTIVE
    transaction (``--last``, resolved through the accepted D1 recent
    operation). Deletion is the accepted D2 idempotent soft delete
    only: there is no hard delete, no restore, and no business
    timezone requirement. One aware UTC clock snapshot becomes the
    mutation ``deleted_at``. On success exactly one JSON object with
    the ``DELETED`` disposition and the transaction ID is printed --
    also for an already DELETED exact target, whose requested final
    state is satisfied and whose original ``deleted_at`` is never
    rewritten; on any failure nothing is printed on stdout and no
    success is claimed.
    """
    command_hint = "transaction delete"
    database_path = _resolve_database_path()
    _require_exactly_one_target(args, command_hint)

    connection = _open_finance_database(database_path)
    try:
        transaction_id = _resolve_transaction_target(connection, args, command_hint)
        mutation_time = _capture_reference_time()
        deleted = soft_delete_transaction(
            connection,
            transaction_id,
            deleted_at=mutation_time,
        )
    finally:
        connection.close()

    outcome = {
        "disposition": "DELETED",
        "transaction_id": deleted.transaction_id,
    }
    print(json.dumps(outcome))
    return 0


def _run_ingest_telegram(args: argparse.Namespace) -> int:
    """Execute ``ingest-telegram``: transport parse, F1 delegation, JSON.

    Constructs only the :class:`~hermes_finance.provenance.TelegramMessageRef`
    provenance and the two parsed transport values, then delegates all
    finance semantics to the accepted stage-F1 facade and prints the
    tiny deterministic JSON outcome on stdout.
    """
    database_path = _resolve_database_path()
    transaction_date = _parse_transaction_date(args.transaction_date)
    received_at = _parse_received_at(args.received_at)
    provenance = TelegramMessageRef(
        chat_id=args.chat_id,
        message_thread_id=args.thread_id,
        message_id=args.message_id,
        update_id=args.update_id,
    )

    connection = _open_finance_database(database_path)
    try:
        result = ingest_finance_message(
            connection,
            args.text,
            provenance,
            transaction_date=transaction_date,
            received_at=received_at,
        )
    finally:
        connection.close()

    outcome = {
        "disposition": result.disposition.name,
        "transaction_id": result.transaction.transaction_id,
    }
    print(json.dumps(outcome))
    return 0


def _run_month(args: argparse.Namespace) -> int:
    """Execute ``month``: F2 delegation with the exact report on stdout.

    Writes exactly the accepted stage-F2 report text to stdout with
    nothing prepended, appended, wrapped, or reformatted. The month is
    either the explicit ``--year`` / ``--month`` pair or the exact
    calendar month resolved from one clock snapshot for
    ``--relative current`` / ``--relative previous``; both alternatives
    feed the identical accepted facade.
    """
    database_path = _resolve_database_path()
    year, month = _resolve_month_period(args, "month")

    connection = _open_finance_database(database_path)
    try:
        report_text = get_monthly_report_text(
            connection,
            year=year,
            month=month,
        )
    finally:
        connection.close()

    sys.stdout.write(report_text)
    return 0


def _run_transactions_recent(args: argparse.Namespace) -> int:
    """Execute ``transactions recent``: G1 delegation, list on stdout.

    ``--limit`` is optional: without it the accepted D1 default limit
    applies. The resolved limit is handed to the accepted G1 facade
    unchanged, and the accepted D1 limit validation stays
    authoritative.
    """
    database_path = _resolve_database_path()
    limit = args.limit if args.limit is not None else DEFAULT_LIMIT

    connection = _open_finance_database(database_path)
    try:
        list_text = get_recent_transactions_text(connection, limit=limit)
    finally:
        connection.close()

    sys.stdout.write(list_text)
    return 0


def _run_transactions_date(args: argparse.Namespace) -> int:
    """Execute ``transactions date``: G1 delegation, list on stdout.

    The period is either an exact ISO ``YYYY-MM-DD`` calendar date
    (``--date``, parsed strictly here as a transport value) or the
    exact business date resolved from one clock snapshot for
    ``--relative today`` / ``--relative yesterday``. Both alternatives
    hand the identical exact date to the accepted G1 facade unchanged.
    """
    database_path = _resolve_database_path()
    transaction_date = _resolve_transactions_date(args)

    connection = _open_finance_database(database_path)
    try:
        list_text = get_transactions_by_date_text(connection, transaction_date)
    finally:
        connection.close()

    sys.stdout.write(list_text)
    return 0


def _run_transactions_month(args: argparse.Namespace) -> int:
    """Execute ``transactions month``: G1 delegation, list on stdout.

    Writes exactly the accepted stage-G1 month-list text to stdout:
    the individual transactions of one exact calendar month, never the
    aggregated monthly report. The month is either the explicit
    ``--year`` / ``--month`` pair or the exact calendar month resolved
    from one clock snapshot for ``--relative current`` /
    ``--relative previous``; both alternatives feed the identical
    accepted facade.
    """
    database_path = _resolve_database_path()
    year, month = _resolve_month_period(args, "transactions month")

    connection = _open_finance_database(database_path)
    try:
        list_text = get_transactions_by_month_text(
            connection,
            year=year,
            month=month,
        )
    finally:
        connection.close()

    sys.stdout.write(list_text)
    return 0


def _run_summary_month(args: argparse.Namespace) -> int:
    """Execute ``summary month``: G2 delegation, summary card on stdout.

    ``--category`` is required; ``--source`` is optional and always
    selects a source inside that category. Both labels are handed to
    the accepted stage-G2 facades unchanged, and the exact card text --
    including the explicit ``no data`` state of a missing label -- is
    written to stdout with nothing prepended, appended, wrapped, or
    reformatted. The month is either the explicit ``--year`` /
    ``--month`` pair or the exact calendar month resolved from one
    clock snapshot for ``--relative current`` / ``--relative
    previous``; the G2 category/source semantics never change.
    """
    database_path = _resolve_database_path()
    year, month = _resolve_month_period(args, "summary month")

    connection = _open_finance_database(database_path)
    try:
        if args.source is None:
            summary_text = get_monthly_category_summary_text(
                connection,
                year=year,
                month=month,
                category=args.category,
            )
        else:
            summary_text = get_monthly_source_summary_text(
                connection,
                year=year,
                month=month,
                category=args.category,
                source=args.source,
            )
    finally:
        connection.close()

    sys.stdout.write(summary_text)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the ``hermes-finance`` argument parser (four command families)."""
    parser = argparse.ArgumentParser(
        prog="hermes-finance",
        description=(
            "Finance-owned CLI boundary over the accepted finance core"
            " (SQLite storage). The database file path is read from the"
            f" {HERMES_FINANCE_DB_PATH} environment variable."
        ),
    )
    subparsers = parser.add_subparsers(
        title="commands", dest="command", required=True, metavar="COMMAND"
    )

    ingest = subparsers.add_parser(
        "ingest-telegram",
        help="ingest one Finance topic message through the accepted F1 intake",
    )
    ingest.add_argument("--text", required=True, help="raw message text")
    ingest.add_argument("--chat-id", required=True, type=int, help="Telegram chat id")
    ingest.add_argument(
        "--message-id", required=True, type=int, help="Telegram message id"
    )
    ingest.add_argument(
        "--update-id", required=True, type=int, help="Telegram update id"
    )
    ingest.add_argument(
        "--transaction-date",
        required=True,
        help="business date as ISO YYYY-MM-DD",
    )
    ingest.add_argument(
        "--received-at",
        required=True,
        help="receive timestamp as ISO-8601 with an explicit UTC offset",
    )
    ingest.add_argument(
        "--thread-id",
        required=True,
        type=int,
        help="Telegram Finance topic thread id (message_thread_id)",
    )

    month = subparsers.add_parser(
        "month",
        help=(
            "print the accepted F2 monthly report for one exact calendar"
            " month (explicit --year/--month or --relative with --timezone)"
        ),
    )
    month.add_argument(
        "--year",
        type=int,
        help="calendar year (together with --month, or use --relative)",
    )
    month.add_argument(
        "--month",
        type=int,
        help="calendar month (together with --year, or use --relative)",
    )
    month.add_argument(
        "--relative",
        choices=sorted(MONTH_PERIODS),
        help="relative month token (requires --timezone)",
    )
    month.add_argument(
        "--timezone",
        help="IANA business timezone name, required with --relative",
    )

    transactions = subparsers.add_parser(
        "transactions",
        help=(
            "show individual stored transactions (recent, one exact date,"
            " or one exact month)"
        ),
    )
    transaction_commands = transactions.add_subparsers(
        title="transaction commands",
        dest="transactions_command",
        required=True,
        metavar="SUBCOMMAND",
    )

    recent = transaction_commands.add_parser(
        "recent",
        help="show the newest transactions (last 10 by default)",
    )
    recent.add_argument(
        "--limit",
        type=int,
        help="number of transactions to show (an int between 1 and 100)",
    )

    transactions_date = transaction_commands.add_parser(
        "date",
        help=(
            "show the transactions of one exact business date"
            " (explicit --date or --relative with --timezone)"
        ),
    )
    transactions_date.add_argument(
        "--date",
        help="business date as ISO YYYY-MM-DD (or use --relative)",
    )
    transactions_date.add_argument(
        "--relative",
        choices=sorted(DATE_PERIODS),
        help="relative date token (requires --timezone)",
    )
    transactions_date.add_argument(
        "--timezone",
        help="IANA business timezone name, required with --relative",
    )

    transactions_month = transaction_commands.add_parser(
        "month",
        help=(
            "show the individual transactions of one exact calendar month"
            " (explicit --year/--month or --relative with --timezone)"
        ),
    )
    transactions_month.add_argument(
        "--year",
        type=int,
        help="calendar year (together with --month, or use --relative)",
    )
    transactions_month.add_argument(
        "--month",
        type=int,
        help="calendar month (together with --year, or use --relative)",
    )
    transactions_month.add_argument(
        "--relative",
        choices=sorted(MONTH_PERIODS),
        help="relative month token (requires --timezone)",
    )
    transactions_month.add_argument(
        "--timezone",
        help="IANA business timezone name, required with --relative",
    )

    summary = subparsers.add_parser(
        "summary",
        help=(
            "show aggregated filtered summaries of one exact calendar"
            " month (one category, or one category and source)"
        ),
    )
    summary_commands = summary.add_subparsers(
        title="summary commands",
        dest="summary_command",
        required=True,
        metavar="SUBCOMMAND",
    )

    summary_month = summary_commands.add_parser(
        "month",
        help=(
            "show the aggregated totals of one exact category (optionally"
            " one exact source inside it) of one exact calendar month"
            " (explicit --year/--month or --relative with --timezone)"
        ),
    )
    summary_month.add_argument(
        "--year",
        type=int,
        help="calendar year (together with --month, or use --relative)",
    )
    summary_month.add_argument(
        "--month",
        type=int,
        help="calendar month (together with --year, or use --relative)",
    )
    summary_month.add_argument(
        "--relative",
        choices=sorted(MONTH_PERIODS),
        help="relative month token (requires --timezone)",
    )
    summary_month.add_argument(
        "--timezone",
        help="IANA business timezone name, required with --relative",
    )
    summary_month.add_argument(
        "--category",
        required=True,
        help="exact category label as persisted in the finance ledger",
    )
    summary_month.add_argument(
        "--source",
        help="exact source label inside the category as persisted in the ledger",
    )

    transaction = subparsers.add_parser(
        "transaction",
        help=(
            "mutate one stored transaction (edit its amount or soft-delete"
            " it; exact --id or --last target)"
        ),
    )
    transaction_commands = transaction.add_subparsers(
        title="transaction commands",
        dest="transaction_command",
        required=True,
        metavar="SUBCOMMAND",
    )

    transaction_edit_amount = transaction_commands.add_parser(
        "edit-amount",
        help=(
            "replace only the amount of one transaction (exact --id or"
            " --last); the stored income/expense direction is preserved"
        ),
    )
    transaction_edit_amount.add_argument(
        "--id",
        help="exact persisted transaction ID (or use --last)",
    )
    transaction_edit_amount.add_argument(
        "--last",
        action="store_true",
        help=(
            "target the latest ACTIVE transaction (the same semantics as"
            " transactions recent --limit 1)"
        ),
    )
    transaction_edit_amount.add_argument(
        "--amount",
        required=True,
        help=(
            "the new positive amount magnitude; the accepted domain"
            " amount validation is authoritative"
        ),
    )

    transaction_delete = transaction_commands.add_parser(
        "delete",
        help=(
            "soft-delete one transaction (exact --id or --last);"
            " idempotent, with no hard delete"
        ),
    )
    transaction_delete.add_argument(
        "--id",
        help="exact persisted transaction ID (or use --last)",
    )
    transaction_delete.add_argument(
        "--last",
        action="store_true",
        help=(
            "target the latest ACTIVE transaction (the same semantics as"
            " transactions recent --limit 1)"
        ),
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``hermes-finance`` CLI and return the process exit status.

    ``argv`` defaults to ``sys.argv[1:]``. Command-line usage errors
    exit with status 2 through :mod:`argparse`; deterministic CLI
    boundary failures (:class:`FinanceCliError`) are reported on stderr
    and exit with status 1; accepted finance-layer failures propagate
    unchanged. Successful runs exit with status 0.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "ingest-telegram":
            return _run_ingest_telegram(args)
        if args.command == "transactions":
            if args.transactions_command == "recent":
                return _run_transactions_recent(args)
            if args.transactions_command == "date":
                return _run_transactions_date(args)
            return _run_transactions_month(args)
        if args.command == "transaction":
            if args.transaction_command == "edit-amount":
                return _run_transaction_edit_amount(args)
            return _run_transaction_delete(args)
        if args.command == "summary":
            return _run_summary_month(args)
        return _run_month(args)
    except FinanceCliError as error:
        print(f"hermes-finance: {error}", file=sys.stderr)
        return 1
