"""Tests for the high-level integration facade (stages F1 and F2).

All tests are deterministic: no network, no wall-clock time, no
randomness, no environment access, no threads or processes. The facade
is exercised through the real accepted parser, creation core, ingest
service, SQLite schema, report aggregation, and renderer; no
lower-layer business logic is re-implemented or stubbed here.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

import pytest

import hermes_finance
import hermes_finance.integration as integration_module
from hermes_finance import (
    Direction,
    FinanceConfig,
    TelegramMessageIdentity,
    TelegramMessageRef,
    Transaction,
    TransactionStatus,
    edit_transaction,
    get_monthly_report_text,
    get_recent_transactions_text,
    get_transactions_by_date_text,
    get_transactions_by_month_text,
    ingest_finance_message,
    open_database,
    render_recent_transactions,
    render_transactions_by_date,
    render_transactions_by_month,
    resolve_relative_date,
    resolve_relative_month,
    soft_delete_transaction,
)
from hermes_finance.ingest import IngestConsistencyError, IngestDisposition
from hermes_finance.operations import (
    list_recent_transactions,
    list_transactions_by_date,
    list_transactions_by_month,
)
from hermes_finance.parser import TransactionParseError
from hermes_finance.repository import (
    RepositoryDataError,
    RepositoryTransactionError,
    find_transaction_by_message,
    persist_transaction,
)

INTEGRATION_SOURCE: str = Path(integration_module.__file__).read_text(encoding="utf-8")

# A fixed non-UTC offset: proves the facade stores the caller timestamp
# exactly, with its original offset, and never converts timezones.
NON_UTC_TZ = timezone(timedelta(hours=3))

FIXED_DATE = date(2026, 9, 3)
FIXED_RECEIVED_AT = datetime(2026, 9, 3, 14, 30, 0, tzinfo=NON_UTC_TZ)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def memory_config() -> FinanceConfig:
    """A valid config pointing at a fresh in-memory database."""
    return FinanceConfig(database_path=Path(":memory:"), business_timezone=NON_UTC_TZ)


def open_memory() -> sqlite3.Connection:
    """Open a fresh migrated in-memory database connection."""
    return open_database(memory_config())


def make_ref(
    *,
    chat_id: int = -100,
    message_thread_id: int = 7,
    message_id: int = 50,
    update_id: int = 1000,
) -> TelegramMessageRef:
    """A valid Telegram message provenance reference."""
    return TelegramMessageRef(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        message_id=message_id,
        update_id=update_id,
    )


def ingest_message(
    conn: sqlite3.Connection,
    text: str,
    *,
    ref: TelegramMessageRef | None = None,
    transaction_date: Any = FIXED_DATE,
    received_at: Any = FIXED_RECEIVED_AT,
) -> Any:
    """ingest_finance_message with the fixed test values and overrides."""
    return ingest_finance_message(
        conn,
        text,
        ref if ref is not None else make_ref(),
        transaction_date=transaction_date,
        received_at=received_at,
    )


def count_transactions(conn: sqlite3.Connection) -> int:
    """Number of transactions rows."""
    row = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()
    return int(row[0])


def count_processed_updates(conn: sqlite3.Connection) -> int:
    """Number of processed_updates rows."""
    row = conn.execute("SELECT COUNT(*) FROM processed_updates").fetchone()
    return int(row[0])


def processed_update_row(conn: sqlite3.Connection, update_id: int) -> Any:
    """The full processed_updates row for the given update_id, or None."""
    return conn.execute(
        "SELECT * FROM processed_updates WHERE update_id = ?", (update_id,)
    ).fetchone()


def transaction_row(conn: sqlite3.Connection) -> Any:
    """The single transactions row (business fields only), or None."""
    return conn.execute(
        "SELECT direction, amount_usdt, category, source, comment,"
        " transaction_date, created_at, updated_at, status, deleted_at"
        " FROM transactions"
    ).fetchone()


def insert_raw_processed_update(
    conn: sqlite3.Connection,
    *,
    update_id: int = 1000,
    chat_id: Any = -100,
    message_thread_id: Any = 7,
    message_id: Any = 50,
    processed_at: str = "2026-09-01T09:00:00+00:00",
) -> None:
    """Insert one raw processed_updates row (bypassing the repository)."""
    conn.execute(
        "INSERT INTO processed_updates ("
        " update_id, chat_id, message_thread_id, message_id, processed_at"
        ") VALUES (?, ?, ?, ?, ?)",
        (update_id, chat_id, message_thread_id, message_id, processed_at),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# CREATED: success contract through the whole composition
# ---------------------------------------------------------------------------


def test_created_income_success_contract() -> None:
    conn = open_memory()
    try:
        result = ingest_message(conn, "+25 Работа Проект A")

        assert result.disposition is IngestDisposition.CREATED

        transaction = result.transaction
        assert transaction.direction is Direction.INCOME
        assert transaction.amount_usdt == Decimal(25)
        assert transaction.category == "Работа"
        assert transaction.source == "Проект A"
        assert transaction.comment is None
        assert transaction.transaction_date == FIXED_DATE
        assert transaction.created_at == FIXED_RECEIVED_AT
        assert transaction.updated_at == FIXED_RECEIVED_AT
        assert transaction.transaction_id == "1"
        assert transaction.status is TransactionStatus.ACTIVE
        assert transaction.deleted_at is None

        assert count_transactions(conn) == 1
        assert count_processed_updates(conn) == 1
        assert conn.in_transaction is False

        row = transaction_row(conn)
        assert row is not None
        assert row == (
            "income",
            "25",
            "Работа",
            "Проект A",
            None,
            "2026-09-03",
            FIXED_RECEIVED_AT.isoformat(),
            FIXED_RECEIVED_AT.isoformat(),
            "active",
            None,
        )

        update = processed_update_row(conn, update_id=1000)
        assert update is not None
        assert update == (1000, -100, 7, 50, FIXED_RECEIVED_AT.isoformat())
    finally:
        conn.close()


def test_created_expense_with_comment_flows_through() -> None:
    conn = open_memory()
    try:
        result = ingest_message(conn, "-10 Инфраструктура Хостинг | сентябрь")

        assert result.disposition is IngestDisposition.CREATED
        assert result.transaction.direction is Direction.EXPENSE
        assert result.transaction.amount_usdt == Decimal(10)
        assert result.transaction.category == "Инфраструктура"
        assert result.transaction.source == "Хостинг"
        assert result.transaction.comment == "сентябрь"
        assert count_transactions(conn) == 1
        assert count_processed_updates(conn) == 1
    finally:
        conn.close()


def test_created_multiword_source_flows_through() -> None:
    conn = open_memory()
    try:
        result = ingest_message(conn, "+25 Работа Проект Бета VIP")

        assert result.disposition is IngestDisposition.CREATED
        assert result.transaction.direction is Direction.INCOME
        assert result.transaction.amount_usdt == Decimal(25)
        assert result.transaction.category == "Работа"
        assert result.transaction.source == "Проект Бета VIP"
        assert result.transaction.comment is None
        assert count_transactions(conn) == 1
    finally:
        conn.close()


def test_created_reuses_received_at_for_created_updated_and_processed() -> None:
    conn = open_memory()
    try:
        received_at = datetime(2026, 10, 31, 23, 59, 59, tzinfo=NON_UTC_TZ)
        result = ingest_message(
            conn, "+1 Кофе Кофейня", received_at=received_at
        )

        assert result.transaction.created_at == received_at
        assert result.transaction.updated_at == received_at

        row = transaction_row(conn)
        assert row is not None
        assert row[6] == received_at.isoformat()
        assert row[7] == received_at.isoformat()

        update = processed_update_row(conn, update_id=1000)
        assert update is not None
        assert update[4] == received_at.isoformat()
    finally:
        conn.close()


def test_created_business_date_is_authoritative_and_independent_of_received_at() -> None:
    conn = open_memory()
    try:
        # The caller deliberately supplies a business date that differs
        # from the calendar date represented by received_at: the facade
        # must use the supplied business date exactly and never derive
        # it from the receive timestamp.
        business_date = date(2026, 9, 2)
        received_at = datetime(2026, 9, 3, 0, 30, 0, tzinfo=NON_UTC_TZ)
        assert business_date != received_at.date()

        result = ingest_message(
            conn, "+25 Работа Проект A", transaction_date=business_date, received_at=received_at
        )

        assert result.disposition is IngestDisposition.CREATED
        assert result.transaction.transaction_date == business_date
        assert result.transaction.transaction_date != received_at.date()

        persisted = find_transaction_by_message(
            conn, TelegramMessageIdentity(chat_id=-100, message_id=50)
        )
        assert persisted is not None
        assert persisted.transaction_date == business_date
        assert persisted.created_at == received_at
        assert persisted.updated_at == received_at

        update = processed_update_row(conn, update_id=1000)
        assert update is not None
        assert update[4] == received_at.isoformat()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# DUPLICATE_MESSAGE
# ---------------------------------------------------------------------------


def test_duplicate_message_returns_original_persisted_transaction() -> None:
    conn = open_memory()
    try:
        first = ingest_message(conn, "+25 Работа Проект A")
        assert first.disposition is IngestDisposition.CREATED

        result = ingest_message(
            conn,
            "+999 Другое ДругойИсточник",
            ref=make_ref(update_id=1001),
        )

        assert result.disposition is IngestDisposition.DUPLICATE_MESSAGE
        assert result.transaction == first.transaction
        assert result.transaction.amount_usdt == Decimal(25)
        assert result.transaction.category == "Работа"
        assert result.transaction.source == "Проект A"
        assert result.transaction.direction is Direction.INCOME

        assert count_transactions(conn) == 1
        assert count_processed_updates(conn) == 2

        row = transaction_row(conn)
        assert row is not None
        assert row[0] == "income"
        assert row[1] == "25"
        assert row[2] == "Работа"
        assert row[3] == "Проект A"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# DUPLICATE_UPDATE
# ---------------------------------------------------------------------------


def test_replay_original_update_is_duplicate_update_without_writes() -> None:
    conn = open_memory()
    try:
        first = ingest_message(conn, "+25 Работа Проект A")
        before_transactions = conn.execute(
            "SELECT * FROM transactions ORDER BY id"
        ).fetchall()
        before_updates = conn.execute(
            "SELECT * FROM processed_updates ORDER BY update_id"
        ).fetchall()

        result = ingest_message(conn, "+25 Работа Проект A")

        assert result.disposition is IngestDisposition.DUPLICATE_UPDATE
        assert result.transaction == first.transaction
        assert conn.in_transaction is False
        assert conn.execute("SELECT * FROM transactions ORDER BY id").fetchall() == (
            before_transactions
        )
        assert conn.execute(
            "SELECT * FROM processed_updates ORDER BY update_id"
        ).fetchall() == before_updates
    finally:
        conn.close()


def test_replay_duplicate_message_update_returns_original_transaction() -> None:
    conn = open_memory()
    try:
        first = ingest_message(conn, "+25 Работа Проект A")
        assert first.disposition is IngestDisposition.CREATED

        duplicate = ingest_message(
            conn, "+25 Работа Проект A", ref=make_ref(update_id=1001)
        )
        assert duplicate.disposition is IngestDisposition.DUPLICATE_MESSAGE

        replay = ingest_message(
            conn, "+25 Работа Проект A", ref=make_ref(update_id=1001)
        )

        assert replay.disposition is IngestDisposition.DUPLICATE_UPDATE
        assert replay.transaction == first.transaction
        assert replay.transaction.transaction_id == first.transaction.transaction_id
        assert count_transactions(conn) == 1
        assert count_processed_updates(conn) == 2
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# validation propagation (accepted lower layers stay authoritative)
# ---------------------------------------------------------------------------


def test_parse_error_propagates_unchanged_without_writes() -> None:
    conn = open_memory()
    try:
        with pytest.raises(TransactionParseError):
            ingest_message(conn, "текст без знака и суммы")

        assert count_transactions(conn) == 0
        assert count_processed_updates(conn) == 0
    finally:
        conn.close()


@pytest.mark.parametrize("bad_text", [None, 5, b"+25 sample text", object()])
def test_non_string_text_rejected_by_parser_boundary(bad_text: Any) -> None:
    conn = open_memory()
    try:
        with pytest.raises(TransactionParseError):
            ingest_message(conn, bad_text)
        assert count_transactions(conn) == 0
        assert count_processed_updates(conn) == 0
    finally:
        conn.close()


def test_datetime_as_transaction_date_rejected_by_domain() -> None:
    conn = open_memory()
    try:
        with pytest.raises(TypeError):
            ingest_message(
                conn,
                "+25 Работа Проект A",
                transaction_date=datetime(2026, 9, 3, 0, 0, 0, tzinfo=NON_UTC_TZ),
            )
        assert count_transactions(conn) == 0
        assert count_processed_updates(conn) == 0
    finally:
        conn.close()


def test_naive_received_at_rejected_by_domain() -> None:
    conn = open_memory()
    try:
        naive = datetime(2026, 9, 3, 14, 30, 0)  # noqa: DTZ001 - naive is the point
        with pytest.raises(ValueError):
            ingest_message(conn, "+25 Работа Проект A", received_at=naive)
        assert count_transactions(conn) == 0
        assert count_processed_updates(conn) == 0
    finally:
        conn.close()


@pytest.mark.parametrize(
    "bad_provenance",
    [
        None,
        "ref",
        5,
        {"chat_id": -100, "message_thread_id": 7, "message_id": 50, "update_id": 1000},
        (-100, 7, 50, 1000),
        object(),
    ],
)
def test_wrong_provenance_type_rejected_by_lower_layer(bad_provenance: Any) -> None:
    conn = open_memory()
    try:
        with pytest.raises(TypeError):
            ingest_finance_message(
                conn,
                "+25 Работа Проект A",
                bad_provenance,
                transaction_date=FIXED_DATE,
                received_at=FIXED_RECEIVED_AT,
            )
        assert count_transactions(conn) == 0
        assert count_processed_updates(conn) == 0
    finally:
        conn.close()


@pytest.mark.parametrize("bad_connection", ["not-a-connection", None, 5, object()])
def test_wrong_connection_type_fails_in_lower_layer(bad_connection: Any) -> None:
    with pytest.raises(TypeError):
        ingest_message(bad_connection, "+25 Работа Проект A")


# ---------------------------------------------------------------------------
# caller-active transaction ownership
# ---------------------------------------------------------------------------


def test_caller_active_transaction_rejected_before_repository_write() -> None:
    conn = open_memory()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO processed_updates ("
            " update_id, chat_id, message_thread_id, message_id, processed_at"
            ") VALUES (?, ?, ?, ?, ?)",
            (1008, -100, 7, 58, "2026-09-01T09:00:00+00:00"),
        )
        assert conn.in_transaction is True

        with pytest.raises(RepositoryTransactionError):
            ingest_message(
                conn, "+25 Работа Проект A", ref=make_ref(update_id=1009)
            )

        # The facade neither committed nor rolled back the caller's
        # transaction: it is still open and the caller's row is still
        # pending inside it.
        assert conn.in_transaction is True
        assert count_processed_updates(conn) == 1
        assert count_transactions(conn) == 0

        conn.rollback()
        assert count_processed_updates(conn) == 0
        assert count_transactions(conn) == 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# consistency error propagation
# ---------------------------------------------------------------------------


def test_conflicting_persisted_provenance_raises_consistency_error() -> None:
    conn = open_memory()
    try:
        ingest_message(conn, "+25 Работа Проект A")

        with pytest.raises(IngestConsistencyError):
            ingest_message(
                conn,
                "+25 Работа Проект A",
                ref=make_ref(chat_id=-20, message_id=8, update_id=1000),
            )

        assert count_transactions(conn) == 1
        assert count_processed_updates(conn) == 1
    finally:
        conn.close()


def test_orphan_processed_update_raises_consistency_error() -> None:
    conn = open_memory()
    try:
        insert_raw_processed_update(
            conn, update_id=3000, chat_id=-100, message_thread_id=7, message_id=77
        )

        with pytest.raises(IngestConsistencyError):
            ingest_message(
                conn, "+25 Работа Проект A", ref=make_ref(message_id=77, update_id=3000)
            )

        assert count_transactions(conn) == 0
        assert count_processed_updates(conn) == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# input immutability
# ---------------------------------------------------------------------------


def test_inputs_are_not_mutated() -> None:
    conn = open_memory()
    try:
        text = "  +25 Работа Проект A  "
        ref = make_ref()

        result = ingest_message(conn, text, ref=ref)

        assert text == "  +25 Работа Проект A  "
        assert ref == make_ref()
        assert FIXED_DATE == date(2026, 9, 3)
        assert FIXED_RECEIVED_AT == datetime(2026, 9, 3, 14, 30, 0, tzinfo=NON_UTC_TZ)
        assert result.transaction is not None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# source hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    [
        "os.environ",
        "getenv",
        "datetime.now",
        "utcnow",
        "time.time",
        "date.today",
        "today()",
        "perf_counter",
        "monotonic",
        "sqlite3.connect",
        "PRAGMA",
        "CREATE TABLE",
        "INSERT INTO",
        "SELECT",
        "DELETE FROM",
        ".execute",
        "import os",
        "import time",
        "import telegram",
        "python-telegram-bot",
        "from telegram",
        "find_transaction_by_update",
        "Transaction(",
        "MonthlyFinanceReport(",
        "Decimal(",
        "sum(",
        "localcontext",
        "Финансы",
        "Доход",
        "Расход",
        "Итог",
        "Категории",
        "USDT",
        "📊",
        "4096",
        "parse_mode",
        "send_message",
        "Добавлено",
        "Уже обработано",
    ],
)
def test_integration_source_avoids_forbidden_operations(forbidden: str) -> None:
    assert forbidden not in INTEGRATION_SOURCE


def test_integration_module_imports_only_the_composed_layers() -> None:
    forbidden_imports = [
        "from hermes_finance.domain import",
        "from hermes_finance.config import",
        "from hermes_finance.storage import",
        "from hermes_finance.repository import",
        "from hermes_finance.mutations import",
    ]
    for forbidden_import in forbidden_imports:
        assert forbidden_import not in INTEGRATION_SOURCE
    composed = {
        "parse_transaction_input": "hermes_finance.parser",
        "create_transaction": "hermes_finance.ledger",
        "ingest_transaction": "hermes_finance.ingest",
        "TelegramMessageRef": "hermes_finance.provenance",
        "build_monthly_report": "hermes_finance.reporting",
        "render_monthly_report": "hermes_finance.rendering",
        "list_recent_transactions": "hermes_finance.operations",
        "list_transactions_by_date": "hermes_finance.operations",
        "list_transactions_by_month": "hermes_finance.operations",
        "render_recent_transactions": "hermes_finance.rendering",
        "render_transactions_by_date": "hermes_finance.rendering",
        "render_transactions_by_month": "hermes_finance.rendering",
        "select_monthly_category": "hermes_finance.filtered_summary",
        "select_category_source": "hermes_finance.filtered_summary",
        "render_monthly_category_summary": "hermes_finance.rendering",
        "render_monthly_source_summary": "hermes_finance.rendering",
    }
    for attribute, expected_module in composed.items():
        assert getattr(integration_module, attribute).__module__ == expected_module


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def test_integration_public_api_is_small_and_deliberate() -> None:
    assert integration_module.__all__ == [
        "get_monthly_category_summary_text",
        "get_monthly_report_text",
        "get_monthly_source_summary_text",
        "get_recent_transactions_text",
        "get_transactions_by_date_text",
        "get_transactions_by_month_text",
        "ingest_finance_message",
    ]
    for name in integration_module.__all__:
        assert callable(getattr(integration_module, name))


def test_package_exports_integration_facade_and_keeps_previous_exports() -> None:
    assert hermes_finance.ingest_finance_message is ingest_finance_message
    assert hermes_finance.get_monthly_report_text is get_monthly_report_text
    assert hermes_finance.get_recent_transactions_text is get_recent_transactions_text
    assert hermes_finance.get_transactions_by_date_text is get_transactions_by_date_text
    assert hermes_finance.get_transactions_by_month_text is get_transactions_by_month_text
    # G3: the pure relative-period resolver is exported too.
    assert hermes_finance.resolve_relative_date is resolve_relative_date
    assert hermes_finance.resolve_relative_month is resolve_relative_month
    assert hermes_finance.__all__ == [
        "DATE_PERIODS",
        "MONTH_PERIODS",
        "SCHEMA_VERSION",
        "CategoryReport",
        "DatabaseMigrationError",
        "Direction",
        "FinanceConfig",
        "IngestConsistencyError",
        "IngestDisposition",
        "IngestResult",
        "MonthlyFinanceReport",
        "ParsedTransactionInput",
        "RepositoryDataError",
        "RepositoryTransactionError",
        "SourceReport",
        "TelegramMessageIdentity",
        "TelegramMessageRef",
        "TelegramUpdateIdentity",
        "Transaction",
        "TransactionMutationError",
        "TransactionNotActiveError",
        "TransactionNotFoundError",
        "TransactionParseError",
        "TransactionStatus",
        "build_monthly_report",
        "create_transaction",
        "edit_transaction",
        "edit_transaction_amount",
        "find_transaction_by_message",
        "find_transaction_by_update",
        "get_monthly_category_summary_text",
        "get_monthly_report_text",
        "get_monthly_source_summary_text",
        "get_processed_update_ref",
        "get_recent_transactions_text",
        "get_transaction",
        "get_transactions_by_date_text",
        "get_transactions_by_month_text",
        "ingest_finance_message",
        "ingest_transaction",
        "is_update_processed",
        "list_recent_transactions",
        "list_transactions_by_date",
        "list_transactions_by_month",
        "normalize_optional_text",
        "normalize_required_text",
        "normalize_usdt_amount",
        "open_database",
        "parse_transaction_input",
        "persist_transaction",
        "record_processed_update",
        "render_monthly_category_summary",
        "render_monthly_report",
        "render_monthly_source_summary",
        "render_recent_transactions",
        "render_transactions_by_date",
        "render_transactions_by_month",
        "require_aware_datetime",
        "require_calendar_date",
        "resolve_relative_date",
        "resolve_relative_month",
        "select_category_source",
        "select_monthly_category",
        "soft_delete_transaction",
    ]


# ---------------------------------------------------------------------------
# F2: get_monthly_report_text -- DB -> E1 -> E2 -> str
# ---------------------------------------------------------------------------

#: The accepted E2 canonical rendering of the six-transaction month.
CANONICAL_AUGUST_REPORT: Final[str] = (
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

#: The accepted E2 empty-month rendering for August 2026.
AUGUST_EMPTY_REPORT: Final[str] = (
    "💰 FINANCE | АВГУСТ 2026\n"
    "\n"
    "📈 Доход: 0 USDT\n"
    "📉 Расход: 0 USDT\n"
    "⚖️ Итог: 0 USDT\n"
    "🧾 Операций: 0\n"
    "\n"
    "📭 Операций за месяц нет."
)


def ingest_august_message(
    conn: sqlite3.Connection,
    text: str,
    *,
    message_id: int,
) -> Any:
    """Ingest one August 2026 Finance message with unique provenance."""
    return ingest_finance_message(
        conn,
        text,
        make_ref(message_id=message_id, update_id=1000 + message_id),
        transaction_date=date(2026, 8, 15),
        received_at=FIXED_RECEIVED_AT,
    )


def insert_raw_august_transaction(
    conn: sqlite3.Connection,
    *,
    message_id: int,
    amount_usdt: Any,
    transaction_date: str = "2026-08-05",
) -> None:
    """Insert one raw ACTIVE August 2026 transactions row.

    Bypasses the domain layer deliberately (for corrupt-row and
    caller-pending-state scenarios); does not commit, so the caller
    owns the transaction state.
    """
    conn.execute(
        "INSERT INTO transactions ("
        " direction, amount_usdt, category, source, comment,"
        " transaction_date, created_at, updated_at, status, deleted_at,"
        " chat_id, message_thread_id, message_id, update_id"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "income",
            amount_usdt,
            "Работа",
            "Проект A",
            None,
            transaction_date,
            FIXED_RECEIVED_AT.isoformat(),
            FIXED_RECEIVED_AT.isoformat(),
            "active",
            None,
            -100,
            7,
            message_id,
            1000 + message_id,
        ),
    )


def test_f2_canonical_six_transaction_month_exact_text() -> None:
    """DB -> E1 -> E2 -> F2 str with exact full-string equality.

    The canonical August 2026 month (six transactions: income 60,
    expense 21, net 39, count 6) is created through the real accepted
    F1 intake into real SQLite persistence, then read back through
    ONLY the F2 operation; the result must equal the accepted E2
    canonical text exactly.
    """
    conn = open_memory()
    try:
        canonical_messages = [
            "+25 Работа Проект A",
            "+30 Работа Проект B",
            "+5 Сервисы Подписка",
            "-10 Инфраструктура Хостинг",
            "-1 Инфраструктура Сервер A",
            "-10 Инфраструктура Сервер B",
        ]
        for message_id, text in enumerate(canonical_messages, start=50):
            result = ingest_august_message(conn, text, message_id=message_id)
            assert result.disposition is IngestDisposition.CREATED

        report_text = get_monthly_report_text(conn, year=2026, month=8)

        assert isinstance(report_text, str)
        assert report_text == CANONICAL_AUGUST_REPORT
        assert not report_text.endswith("\n")
    finally:
        conn.close()


def test_f2_empty_month_exact_text() -> None:
    """A valid empty month renders the accepted E2 empty-month text.

    No F2-specific special case is involved: the empty report flows
    from E1 through E2 unchanged.
    """
    conn = open_memory()
    try:
        report_text = get_monthly_report_text(conn, year=2026, month=5)

        assert report_text == (
            "💰 FINANCE | МАЙ 2026\n"
            "\n"
            "📈 Доход: 0 USDT\n"
            "📉 Расход: 0 USDT\n"
            "⚖️ Итог: 0 USDT\n"
            "🧾 Операций: 0\n"
            "\n"
            "📭 Операций за месяц нет."
        )
    finally:
        conn.close()


def test_f2_reflects_edited_current_state_then_soft_delete_excludes() -> None:
    """D2 current-state integration: edit then soft-delete through F2.

    The F2 report must reflect the edited current state of the
    transaction (full replacement semantics of the accepted D2 edit),
    and after the accepted D2 soft delete the subsequent F2 report
    must exclude the transaction entirely.
    """
    conn = open_memory()
    try:
        created = ingest_august_message(conn, "+25 Работа Проект A", message_id=60)
        transaction_id = created.transaction.transaction_id or ""
        assert "Доход: 25 USDT" in get_monthly_report_text(conn, year=2026, month=8)

        edit_transaction(
            conn,
            transaction_id,
            direction=Direction.EXPENSE,
            amount_usdt=Decimal(40),
            category="Инфраструктура",
            source="Сервер B",
            comment=None,
            transaction_date=date(2026, 8, 20),
            updated_at=FIXED_RECEIVED_AT,
        )

        assert get_monthly_report_text(conn, year=2026, month=8) == (
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

        soft_delete_transaction(
            conn, transaction_id, deleted_at=FIXED_RECEIVED_AT
        )

        assert get_monthly_report_text(conn, year=2026, month=8) == AUGUST_EMPTY_REPORT
    finally:
        conn.close()


def test_f2_idempotent_redelivery_contributes_exactly_once() -> None:
    """C3 integration: CREATED then DUPLICATE_MESSAGE/DUPLICATE_UPDATE.

    After an idempotent redelivery sequence the F2 report must show
    exactly one financial contribution; F2 itself knows nothing about
    processed_updates and never re-implements C3 logic.
    """
    conn = open_memory()
    try:
        first = ingest_august_message(conn, "+25 Работа Проект A", message_id=61)
        assert first.disposition is IngestDisposition.CREATED

        duplicate = ingest_finance_message(
            conn,
            "+999 Другое ДругойИсточник",
            make_ref(message_id=61, update_id=2000),
            transaction_date=date(2026, 8, 15),
            received_at=FIXED_RECEIVED_AT,
        )
        assert duplicate.disposition is IngestDisposition.DUPLICATE_MESSAGE

        replay = ingest_august_message(conn, "+25 Работа Проект A", message_id=61)
        assert replay.disposition is IngestDisposition.DUPLICATE_UPDATE

        report_text = get_monthly_report_text(conn, year=2026, month=8)

        assert "Доход: 25 USDT" in report_text
        assert "999" not in report_text
        assert "Операций: 1" in report_text
        assert "📂 Работа" in report_text
        assert count_transactions(conn) == 1
    finally:
        conn.close()


def test_f2_exact_arbitrary_precision_amount_passthrough() -> None:
    """An arbitrary-precision domain amount survives to the exact text.

    A high-precision amount accepted by the domain is persisted through
    the accepted repository and must appear in the F2 output with the
    exact E1/E2 money representation: no rounding, no scientific
    notation, no truncation.
    """
    conn = open_memory()
    try:
        amount = Decimal("999999999999999999.999999999999999999")
        transaction = Transaction(
            direction=Direction.INCOME,
            amount_usdt=amount,
            category="Работа",
            source="Проект B",
            comment=None,
            transaction_date=date(2026, 8, 10),
            created_at=FIXED_RECEIVED_AT,
            updated_at=FIXED_RECEIVED_AT,
            status=TransactionStatus.ACTIVE,
            deleted_at=None,
        )
        persist_transaction(
            conn,
            transaction,
            make_ref(message_id=62, update_id=1062),
            processed_at=FIXED_RECEIVED_AT,
        )

        report_text = get_monthly_report_text(conn, year=2026, month=8)

        expected = "999999999999999999.999999999999999999"
        assert f"Доход: {expected} USDT" in report_text
        assert f"Итог: +{expected} USDT" in report_text
        assert "• Проект B" in report_text
        assert f"  доход {expected} · расход 0 · итог +{expected} USDT · операций 1" in report_text
        assert "E+" not in report_text
        assert "E-" not in report_text
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("year", "month"),
    [
        (2026, 0),
        (2026, 13),
        (0, 9),
        (10000, 9),
        (2026, "8"),
        (2026, None),
        (2026, True),
    ],
)
def test_f2_invalid_year_month_propagates_accepted_validation(
    year: Any, month: Any
) -> None:
    """Invalid year/month fails through the accepted lower layer.

    F2 adds no competing validation: the accepted E1/D1 validation
    raises unchanged, no wrapping, and nothing is written.
    """
    conn = open_memory()
    try:
        with pytest.raises((TypeError, ValueError)):
            get_monthly_report_text(conn, year=year, month=month)
        assert count_transactions(conn) == 0
        assert count_processed_updates(conn) == 0
    finally:
        conn.close()


def test_f2_corrupt_row_propagates_repository_data_error() -> None:
    """One representative corrupt-row scenario propagates unchanged.

    A persisted ACTIVE row with a non-decimal amount must make F2 raise
    the accepted RepositoryDataError (with its original cause), never
    swallowing or wrapping it into an empty report.
    """
    conn = open_memory()
    try:
        ingest_august_message(conn, "+25 Работа Проект A", message_id=63)
        insert_raw_august_transaction(
            conn, message_id=64, amount_usdt="not-a-decimal", transaction_date="2026-08-02"
        )
        conn.commit()

        with pytest.raises(RepositoryDataError) as exc_info:
            get_monthly_report_text(conn, year=2026, month=8)

        assert exc_info.value.__cause__ is not None
    finally:
        conn.close()


def test_f2_read_only_caller_transaction_remains_owned_by_caller() -> None:
    """F2 is read-only: the caller's transaction stays the caller's.

    With a caller-active transaction holding a pending row, the F2
    report observes the same connection's visible state (per accepted
    D1 semantics) while neither committing nor rolling back; after the
    caller rolls back, the pending contribution is gone.
    """
    conn = open_memory()
    try:
        conn.execute("BEGIN")
        insert_raw_august_transaction(conn, message_id=65, amount_usdt="7")
        assert conn.in_transaction is True

        report_text = get_monthly_report_text(conn, year=2026, month=8)

        assert "Доход: 7 USDT" in report_text
        assert "Операций: 1" in report_text
        assert conn.in_transaction is True

        conn.rollback()
        assert conn.in_transaction is False

        assert get_monthly_report_text(conn, year=2026, month=8) == AUGUST_EMPTY_REPORT
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# G1: transaction-list facades -- DB -> D1 -> renderer -> str
# ---------------------------------------------------------------------------


def ingest_september_message(
    conn: sqlite3.Connection,
    text: str,
    *,
    message_id: int,
    transaction_date: Any = None,
) -> Any:
    """Ingest one Finance message with unique provenance (Sept 2026 by default)."""
    return ingest_finance_message(
        conn,
        text,
        make_ref(message_id=message_id, update_id=1000 + message_id),
        transaction_date=transaction_date
        if transaction_date is not None
        else date(2026, 9, 5),
        received_at=FIXED_RECEIVED_AT,
    )


def test_g1_recent_facade_matches_accepted_d1_and_renderer_composition() -> None:
    """The facade output equals D1 -> renderer composed by hand."""
    conn = open_memory()
    try:
        ingest_september_message(conn, "+25 Работа Проект A", message_id=70)
        ingest_september_message(
            conn, "-10 Инфраструктура Сервер B | продление сервера", message_id=71
        )

        facade_text = get_recent_transactions_text(conn, limit=10)
        composed_text = render_recent_transactions(
            list_recent_transactions(conn, limit=10)
        )

        assert facade_text == composed_text
        assert facade_text == (
            "🧾 FINANCE | ПОСЛЕДНИЕ ОПЕРАЦИИ\n"
            "\n"
            "🔴 #2 | -10 USDT | 05.09.2026\n"
            "Инфраструктура • Сервер B\n"
            "💬 продление сервера\n"
            "\n"
            "🟢 #1 | +25 USDT | 05.09.2026\n"
            "Работа • Проект A"
        )
    finally:
        conn.close()


def test_g1_date_facade_matches_accepted_d1_and_renderer_composition() -> None:
    conn = open_memory()
    try:
        ingest_september_message(conn, "+25 Работа Проект A", message_id=72)
        ingest_september_message(
            conn, "+5 Сервисы Подписка", message_id=73, transaction_date=date(2026, 9, 4)
        )

        facade_text = get_transactions_by_date_text(conn, date(2026, 9, 4))
        composed_text = render_transactions_by_date(
            list_transactions_by_date(conn, date(2026, 9, 4)),
            date(2026, 9, 4),
        )

        assert facade_text == composed_text
        assert facade_text == (
            "🧾 FINANCE | ОПЕРАЦИИ 04.09.2026\n"
            "\n"
            "🟢 #2 | +5 USDT | 04.09.2026\n"
            "Сервисы • Подписка"
        )
    finally:
        conn.close()


def test_g1_month_facade_matches_accepted_d1_and_renderer_composition() -> None:
    conn = open_memory()
    try:
        ingest_september_message(conn, "+25 Работа Проект A", message_id=74)
        ingest_september_message(
            conn, "+5 Сервисы Подписка", message_id=75, transaction_date=date(2026, 9, 4)
        )

        facade_text = get_transactions_by_month_text(conn, year=2026, month=9)
        composed_text = render_transactions_by_month(
            list_transactions_by_month(conn, year=2026, month=9),
            year=2026,
            month=9,
        )

        assert facade_text == composed_text
        assert facade_text == (
            "🧾 FINANCE | ОПЕРАЦИИ СЕНТЯБРЬ 2026\n"
            "\n"
            "🟢 #1 | +25 USDT | 05.09.2026\n"
            "Работа • Проект A\n"
            "\n"
            "🟢 #2 | +5 USDT | 04.09.2026\n"
            "Сервисы • Подписка"
        )
    finally:
        conn.close()


def test_g1_recent_default_limit_is_ten_newest() -> None:
    """Without an explicit limit the accepted D1 default (10) applies."""
    conn = open_memory()
    try:
        for index in range(12):
            ingest_september_message(
                conn,
                "+1 Работа Проект A",
                message_id=80 + index,
                transaction_date=date(2026, 9, 1 + index),
            )

        text = get_recent_transactions_text(conn)

        card_lines = [line for line in text.split("\n") if line.startswith("🟢")]
        assert len(card_lines) == 10
        # The ten newest (2026-09-12 down to 2026-09-03), newest first.
        assert "12.09.2026" in card_lines[0]
        assert "03.09.2026" in card_lines[-1]
        assert "01.09.2026" not in text
    finally:
        conn.close()


def test_g1_empty_states_for_all_three_facades() -> None:
    conn = open_memory()
    try:
        assert get_recent_transactions_text(conn) == (
            "🧾 FINANCE | ПОСЛЕДНИЕ ОПЕРАЦИИ\n\n📭 Операций нет."
        )
        assert get_transactions_by_date_text(conn, date(2026, 9, 5)) == (
            "🧾 FINANCE | ОПЕРАЦИИ 05.09.2026\n\n📭 Операций нет."
        )
        assert get_transactions_by_month_text(conn, year=2026, month=9) == (
            "🧾 FINANCE | ОПЕРАЦИИ СЕНТЯБРЬ 2026\n\n📭 Операций нет."
        )
    finally:
        conn.close()


def test_g1_active_only_visibility_inherited_from_d1() -> None:
    """Soft-deleted transactions are invisible through every facade."""
    conn = open_memory()
    try:
        created = ingest_september_message(conn, "+25 Работа Проект A", message_id=90)
        ingest_september_message(conn, "+5 Сервисы Подписка", message_id=91)
        soft_delete_transaction(
            conn, created.transaction.transaction_id or "", deleted_at=FIXED_RECEIVED_AT
        )

        assert get_recent_transactions_text(conn) == (
            "🧾 FINANCE | ПОСЛЕДНИЕ ОПЕРАЦИИ\n"
            "\n"
            "🟢 #2 | +5 USDT | 05.09.2026\n"
            "Сервисы • Подписка"
        )
        assert get_transactions_by_date_text(conn, date(2026, 9, 5)) == (
            "🧾 FINANCE | ОПЕРАЦИИ 05.09.2026\n"
            "\n"
            "🟢 #2 | +5 USDT | 05.09.2026\n"
            "Сервисы • Подписка"
        )
        assert get_transactions_by_month_text(conn, year=2026, month=9) == (
            "🧾 FINANCE | ОПЕРАЦИИ СЕНТЯБРЬ 2026\n"
            "\n"
            "🟢 #2 | +5 USDT | 05.09.2026\n"
            "Сервисы • Подписка"
        )
    finally:
        conn.close()


def test_g1_no_resorting_or_reaggregation_order_inherited_from_d1() -> None:
    """The facade displays the exact D1 newest-first sequence."""
    conn = open_memory()
    try:
        ingest_september_message(
            conn, "+25 Работа Проект A", message_id=92, transaction_date=date(2026, 9, 1)
        )
        ingest_september_message(
            conn, "+5 Сервисы Подписка", message_id=93, transaction_date=date(2026, 9, 2)
        )
        ingest_september_message(
            conn, "-10 Инфраструктура Сервер B", message_id=94, transaction_date=date(2026, 9, 2)
        )

        text = get_transactions_by_month_text(conn, year=2026, month=9)
        id_lines = [
            line for line in text.split("\n") if line.startswith(("🟢", "🔴"))
        ]

        # D1 order: transaction_date DESC, id DESC -> 02.09 #3, 02.09 #2, 01.09 #1.
        assert [line.split(" | ")[0] for line in id_lines] == [
            "🔴 #3",
            "🟢 #2",
            "🟢 #1",
        ]
    finally:
        conn.close()


@pytest.mark.parametrize(
    "bad_limit",
    [0, -1, 101, True, "5", 5.0, None],
)
def test_g1_recent_limit_validation_propagates_from_d1(bad_limit: Any) -> None:
    conn = open_memory()
    try:
        with pytest.raises((TypeError, ValueError)):
            get_recent_transactions_text(conn, limit=bad_limit)
        assert count_transactions(conn) == 0
    finally:
        conn.close()


@pytest.mark.parametrize(
    "bad_date",
    [
        datetime(2026, 9, 5, 0, 0, 0, tzinfo=NON_UTC_TZ),
        "2026-09-05",
        None,
        20260905,
    ],
)
def test_g1_date_validation_propagates_from_lower_layer(bad_date: Any) -> None:
    conn = open_memory()
    try:
        with pytest.raises((TypeError, ValueError)):
            get_transactions_by_date_text(conn, bad_date)
        assert count_transactions(conn) == 0
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("year", "month"),
    [
        (2026, 0),
        (2026, 13),
        (0, 9),
        (10000, 9),
        (2026, "9"),
        (2026, None),
        ("2026", 9),
        (True, 9),
    ],
)
def test_g1_month_validation_propagates_from_d1(year: Any, month: Any) -> None:
    conn = open_memory()
    try:
        with pytest.raises((TypeError, ValueError)):
            get_transactions_by_month_text(conn, year=year, month=month)
        assert count_transactions(conn) == 0
    finally:
        conn.close()


def test_g1_corrupt_row_propagates_repository_data_error() -> None:
    conn = open_memory()
    try:
        ingest_september_message(conn, "+25 Работа Проект A", message_id=95)
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
                FIXED_RECEIVED_AT.isoformat(),
                FIXED_RECEIVED_AT.isoformat(),
                "active",
                None,
                -100,
                7,
                96,
                1096,
            ),
        )
        conn.commit()

        with pytest.raises(RepositoryDataError) as exc_info:
            get_recent_transactions_text(conn)
        assert exc_info.value.__cause__ is not None

        with pytest.raises(RepositoryDataError):
            get_transactions_by_date_text(conn, date(2026, 9, 4))
        with pytest.raises(RepositoryDataError):
            get_transactions_by_month_text(conn, year=2026, month=9)
    finally:
        conn.close()


def test_g1_facades_are_read_only() -> None:
    """No facade writes rows or touches the caller's transaction state."""
    conn = open_memory()
    try:
        ingest_september_message(conn, "+25 Работа Проект A", message_id=97)

        conn.execute("BEGIN")
        assert conn.in_transaction is True
        before = conn.execute("SELECT * FROM transactions").fetchall()
        before_updates = conn.execute(
            "SELECT * FROM processed_updates"
        ).fetchall()

        get_recent_transactions_text(conn)
        get_transactions_by_date_text(conn, date(2026, 9, 5))
        get_transactions_by_month_text(conn, year=2026, month=9)

        assert conn.in_transaction is True
        assert conn.execute("SELECT * FROM transactions").fetchall() == before
        assert (
            conn.execute("SELECT * FROM processed_updates").fetchall()
            == before_updates
        )
        conn.rollback()
    finally:
        conn.close()


def test_g1_edited_current_state_flows_through_list_facades() -> None:
    conn = open_memory()
    try:
        created = ingest_september_message(conn, "+25 Работа Проект A", message_id=98)
        edit_transaction(
            conn,
            created.transaction.transaction_id or "",
            direction=Direction.EXPENSE,
            amount_usdt=Decimal(40),
            category="Инфраструктура",
            source="Сервер B",
            comment=None,
            transaction_date=date(2026, 9, 6),
            updated_at=FIXED_RECEIVED_AT,
        )

        assert get_recent_transactions_text(conn) == (
            "🧾 FINANCE | ПОСЛЕДНИЕ ОПЕРАЦИИ\n"
            "\n"
            "🔴 #1 | -40 USDT | 06.09.2026\n"
            "Инфраструктура • Сервер B"
        )
    finally:
        conn.close()
