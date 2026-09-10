"""Tests for the finance-owned CLI boundary (stages F3, G1, G2, G3).

All tests are deterministic: no network, no wall-clock time, no
randomness, no subprocesses, no Hermes host. The CLI is exercised
in-process through :func:`hermes_finance.cli.main` against real
temporary SQLite database files; the accepted F1/F2 semantics behind
the boundary are never re-implemented or stubbed here.

Relative-period invocations (G3) never read the real clock either: the
CLI's single clock snapshot seam is patched with an explicit aware
instant, and the IANA business timezones are provided through the same
bounded tz-database convention the stage-F5 plugin tests established.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import tomllib
import zoneinfo
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Final

import pytest

import hermes_finance.cli as cli_module
from hermes_finance import (
    FinanceConfig,
    TransactionNotActiveError,
    TransactionNotFoundError,
    open_database,
)
from hermes_finance.cli import HERMES_FINANCE_DB_PATH, main
from hermes_finance.parser import TransactionParseError

PYPROJECT_PATH: Final[Path] = Path(__file__).resolve().parent.parent / "pyproject.toml"
CLI_SOURCE: Final[str] = Path(cli_module.__file__).read_text(encoding="utf-8")

RECEIVED_AT_TEXT: Final[str] = "2026-09-03T14:30:00+03:00"
FLOW_DATE_TEXT: Final[str] = "2026-08-05"
FLOW_RECEIVED_AT_TEXT: Final[str] = "2026-08-05T09:15:30+05:00"

#: The finance test venv deliberately ships no IANA tz database (the
#: established stage-F5 test convention), so a bounded mapping provides
#: the identifiers these tests need while every unknown key raises the
#: real ``ZoneInfoNotFoundError``. The offsets are the exact ones the
#: tested periods need: Europe/Moscow is fixed UTC+3, Asia/Bangkok is
#: fixed UTC+7, and America/New_York uses its September daylight
#: offset (UTC-4). The mapping is deliberately exhaustive: no key is
#: ever delegated to the host/system timezone database, so the tests
#: stay deterministic even on hosts whose tzdata resolves extra keys
#: (for example a lowercase "utc").
_BOUNDED_IANA_ZONES: Final[dict[str, timezone]] = {
    "UTC": UTC,
    "Europe/Moscow": timezone(timedelta(hours=3)),
    "Asia/Bangkok": timezone(timedelta(hours=7)),
    "America/New_York": timezone(timedelta(hours=-4)),
}


def _bounded_zoneinfo(key: str) -> tzinfo:
    """Resolve the bounded test identifiers, reject every other key."""
    if key in _BOUNDED_IANA_ZONES:
        return _BOUNDED_IANA_ZONES[key]
    raise zoneinfo.ZoneInfoNotFoundError(f"No time zone found with key {key!r}")


@pytest.fixture
def bounded_tz_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``zoneinfo.ZoneInfo`` provide only the bounded test identifiers.

    The CLI resolves ``--timezone`` through ``zoneinfo.ZoneInfo``
    attribute access at invocation time, so patching the module
    attribute cleanly replaces the lookup: the bounded identifiers
    resolve to their fixed offsets and every unknown key raises
    ``ZoneInfoNotFoundError`` without ever consulting the host or
    system timezone database.
    """
    monkeypatch.setattr(zoneinfo, "ZoneInfo", _bounded_zoneinfo)


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

#: The accepted E2 empty-month rendering for May 2026.
EMPTY_MAY_REPORT: Final[str] = (
    "💰 FINANCE | МАЙ 2026\n"
    "\n"
    "📈 Доход: 0 USDT\n"
    "📉 Расход: 0 USDT\n"
    "⚖️ Итог: 0 USDT\n"
    "🧾 Операций: 0\n"
    "\n"
    "📭 Операций за месяц нет."
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def use_temp_database(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point HERMES_FINANCE_DB_PATH at a fresh temp database file path."""
    database_path = tmp_path / "finance.db"
    monkeypatch.setenv(HERMES_FINANCE_DB_PATH, str(database_path))
    return database_path


def ingest_args(
    text: str = "+25 Работа Проект A",
    *,
    chat_id: int = -100,
    message_id: int = 50,
    update_id: int = 1000,
    thread_id: int = 7,
    transaction_date: str = "2026-09-03",
    received_at: str = RECEIVED_AT_TEXT,
) -> list[str]:
    """Build one ``ingest-telegram`` argument vector.

    ``--thread-id`` is always included: it is a required argument with
    no default, so every successful invocation carries an explicit real
    Telegram topic thread id.
    """
    return [
        "ingest-telegram",
        "--text",
        text,
        "--chat-id",
        str(chat_id),
        "--message-id",
        str(message_id),
        "--update-id",
        str(update_id),
        "--thread-id",
        str(thread_id),
        "--transaction-date",
        transaction_date,
        "--received-at",
        received_at,
    ]


def open_for_inspection(database_path: Path) -> sqlite3.Connection:
    """Open the temp database through the accepted bootstrap."""
    return open_database(
        FinanceConfig(database_path=database_path, business_timezone=UTC)
    )


def transaction_rows(connection: sqlite3.Connection) -> list[Any]:
    """All ``transactions`` rows with full provenance columns."""
    return connection.execute(
        "SELECT direction, amount_usdt, category, source, comment,"
        " transaction_date, created_at, updated_at, status, deleted_at,"
        " chat_id, message_thread_id, message_id, update_id"
        " FROM transactions"
    ).fetchall()


def processed_update_rows(connection: sqlite3.Connection) -> list[Any]:
    """All ``processed_updates`` rows."""
    return connection.execute(
        "SELECT update_id, chat_id, message_thread_id, message_id, processed_at"
        " FROM processed_updates"
    ).fetchall()


# ---------------------------------------------------------------------------
# A/B: console entry point declared, superseded plugin entry point absent
# ---------------------------------------------------------------------------


def test_console_script_entry_point_declared() -> None:
    """pyproject declares hermes-finance -> hermes_finance.cli:main."""
    with PYPROJECT_PATH.open("rb") as stream:
        data = tomllib.load(stream)
    scripts = data["project"].get("scripts", {})
    assert scripts.get("hermes-finance") == "hermes_finance.cli:main"
    assert callable(cli_module.main)


def test_superseded_hermes_plugin_entry_point_absent() -> None:
    """No hermes_agent.plugins entry-point group remains in pyproject."""
    with PYPROJECT_PATH.open("rb") as stream:
        data = tomllib.load(stream)
    entry_point_groups = data["project"].get("entry-points", {})
    assert "hermes_agent.plugins" not in entry_point_groups
    for group in entry_point_groups:
        assert "hermes_agent" not in group


def test_superseded_plugin_module_and_tests_are_gone() -> None:
    """The superseded plugin module and its tests no longer exist."""
    assert importlib.util.find_spec("hermes_finance.hermes_plugin") is None
    project_root = Path(__file__).resolve().parent.parent
    assert not (project_root / "src" / "hermes_finance" / "hermes_plugin.py").exists()
    assert not (project_root / "tests" / "test_hermes_plugin.py").exists()


# ---------------------------------------------------------------------------
# C/D/E: ingest dispositions through real temp SQLite
# ---------------------------------------------------------------------------


def test_ingest_created_through_real_temp_sqlite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)

    exit_code = main(ingest_args())

    assert exit_code == 0
    out = capsys.readouterr().out
    assert json.loads(out) == {"disposition": "CREATED", "transaction_id": "1"}
    assert out.count("\n") == 1

    connection = open_for_inspection(database_path)
    try:
        assert transaction_rows(connection) == [
            (
                "income",
                "25",
                "Работа",
                "Проект A",
                None,
                "2026-09-03",
                RECEIVED_AT_TEXT,
                RECEIVED_AT_TEXT,
                "active",
                None,
                -100,
                7,
                50,
                1000,
            )
        ]
        assert processed_update_rows(connection) == [
            (1000, -100, 7, 50, RECEIVED_AT_TEXT)
        ]
    finally:
        connection.close()


def test_duplicate_update_replays_without_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    use_temp_database(monkeypatch, tmp_path)

    assert main(ingest_args()) == 0
    capsys.readouterr()

    exit_code = main(ingest_args())

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "disposition": "DUPLICATE_UPDATE",
        "transaction_id": "1",
    }

    connection = open_for_inspection(tmp_path / "finance.db")
    try:
        assert len(transaction_rows(connection)) == 1
        assert len(processed_update_rows(connection)) == 1
    finally:
        connection.close()


def test_duplicate_logical_message_new_update(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    use_temp_database(monkeypatch, tmp_path)

    assert main(ingest_args()) == 0
    capsys.readouterr()

    exit_code = main(
        ingest_args("+999 Другое ДругойИсточник", message_id=50, update_id=1001)
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "disposition": "DUPLICATE_MESSAGE",
        "transaction_id": "1",
    }

    connection = open_for_inspection(tmp_path / "finance.db")
    try:
        rows = transaction_rows(connection)
        assert len(rows) == 1
        # The persisted original transaction wins: no second financial
        # transaction is created from the redelivered text.
        assert rows[0][:4] == ("income", "25", "Работа", "Проект A")
        assert len(processed_update_rows(connection)) == 2
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# F: exact transport flow into accepted F1 semantics
# ---------------------------------------------------------------------------


def test_exact_values_flow_into_accepted_f1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)

    exit_code = main(
        ingest_args(
            "+25 Работа Проект A",
            chat_id=-100,
            message_id=777,
            update_id=9001,
            thread_id=12345,
            transaction_date=FLOW_DATE_TEXT,
            received_at=FLOW_RECEIVED_AT_TEXT,
        )
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["transaction_id"] == "1"

    connection = open_for_inspection(database_path)
    try:
        # The representative real Telegram topic thread id 12345
        # reaches TelegramMessageRef and both persisted rows unchanged.
        assert transaction_rows(connection) == [
            (
                "income",
                "25",
                "Работа",
                "Проект A",
                None,
                FLOW_DATE_TEXT,
                FLOW_RECEIVED_AT_TEXT,
                FLOW_RECEIVED_AT_TEXT,
                "active",
                None,
                -100,
                12345,
                777,
                9001,
            )
        ]
        assert processed_update_rows(connection) == [
            (9001, -100, 12345, 777, FLOW_RECEIVED_AT_TEXT)
        ]
    finally:
        connection.close()


def test_missing_thread_id_is_a_usage_error_without_db_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)

    without_thread = [
        argument
        for argument in ingest_args()
        if argument != "--thread-id" and argument != "7"
    ]

    with pytest.raises(SystemExit) as exit_info:
        main(without_thread)
    assert exit_info.value.code == 2
    assert not database_path.exists()


@pytest.mark.parametrize("invalid_thread_id", [0, -5])
def test_invalid_thread_id_propagates_accepted_provenance_validation(
    invalid_thread_id: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The accepted provenance value object stays authoritative.

    A thread id the provenance model rejects raises unchanged through
    the CLI boundary before any database work happens: the CLI never
    adds a competing validation or a substitute value.
    """
    database_path = use_temp_database(monkeypatch, tmp_path)

    with pytest.raises(ValueError):
        main(ingest_args(thread_id=invalid_thread_id))
    assert not database_path.exists()


def test_expense_text_with_leading_minus_is_ingested_verbatim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    use_temp_database(monkeypatch, tmp_path)

    exit_code = main(ingest_args("-10 Инфраструктура Хостинг | сентябрь"))

    assert exit_code == 0
    connection = open_for_inspection(tmp_path / "finance.db")
    try:
        row = transaction_rows(connection)[0]
        assert row[0] == "expense"
        assert row[1] == "10"
        assert row[2] == "Инфраструктура"
        assert row[3] == "Хостинг"
        assert row[4] == "сентябрь"
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# G: invalid finance text propagates deterministically with no write
# ---------------------------------------------------------------------------


def test_invalid_finance_text_propagates_without_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)

    with pytest.raises(TransactionParseError):
        main(ingest_args("текст без знака и суммы"))

    connection = open_for_inspection(database_path)
    try:
        assert transaction_rows(connection) == []
        assert processed_update_rows(connection) == []
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# H: transport parsing failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_date",
    ["2026-9-3", "09/03/2026", "20260903", "2026-02-30", "not-a-date"],
)
def test_invalid_transaction_date_rejected_before_database_work(
    bad_date: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)

    exit_code = main(ingest_args(transaction_date=bad_date))

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "--transaction-date" in captured.err
    assert captured.out == ""
    assert not database_path.exists()


@pytest.mark.parametrize(
    "bad_received_at",
    ["2026-09-03T14:30:00", "not-a-datetime", "2026-09-03T25:00:00+03:00"],
)
def test_invalid_received_at_rejected_before_database_work(
    bad_received_at: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)

    exit_code = main(ingest_args(received_at=bad_received_at))

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "--received-at" in captured.err
    assert captured.out == ""
    assert not database_path.exists()


def test_naive_received_at_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)

    exit_code = main(ingest_args(received_at="2026-09-03T14:30:00"))

    assert exit_code == 1
    assert "timezone-aware" in capsys.readouterr().err
    assert not database_path.exists()


def test_z_suffix_received_at_is_accepted_aware(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    use_temp_database(monkeypatch, tmp_path)

    exit_code = main(ingest_args(received_at="2026-09-03T14:30:00Z"))

    assert exit_code == 0
    connection = open_for_inspection(tmp_path / "finance.db")
    try:
        assert len(transaction_rows(connection)) == 1
        assert transaction_rows(connection)[0][6] == "2026-09-03T14:30:00+00:00"
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# I/J: month command returns exact accepted F2 report text
# ---------------------------------------------------------------------------


def test_month_returns_exact_accepted_f2_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    use_temp_database(monkeypatch, tmp_path)

    canonical_messages = [
        "+25 Работа Проект A",
        "+30 Работа Проект B",
        "+5 Сервисы Подписка",
        "-10 Инфраструктура Хостинг",
        "-1 Инфраструктура Сервер A",
        "-10 Инфраструктура Сервер B",
    ]
    for message_id, text in enumerate(canonical_messages, start=50):
        assert (
            main(
                ingest_args(
                    text,
                    message_id=message_id,
                    update_id=1000 + message_id,
                    transaction_date="2026-08-15",
                )
            )
            == 0
        )
    capsys.readouterr()

    exit_code = main(["month", "--year", "2026", "--month", "8"])

    assert exit_code == 0
    assert capsys.readouterr().out == CANONICAL_AUGUST_REPORT


def test_month_empty_month_exact_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    use_temp_database(monkeypatch, tmp_path)

    exit_code = main(["month", "--year", "2026", "--month", "5"])

    assert exit_code == 0
    assert capsys.readouterr().out == EMPTY_MAY_REPORT


def test_month_invalid_month_propagates_accepted_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_temp_database(monkeypatch, tmp_path)

    with pytest.raises(ValueError):
        main(["month", "--year", "2026", "--month", "13"])


# ---------------------------------------------------------------------------
# K: database environment configuration
# ---------------------------------------------------------------------------


def test_missing_db_env_rejected_for_both_commands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(HERMES_FINANCE_DB_PATH, raising=False)

    assert main(ingest_args()) == 1
    assert main(["month", "--year", "2026", "--month", "8"]) == 1

    errors = capsys.readouterr()
    assert errors.out == ""
    assert errors.err.count(HERMES_FINANCE_DB_PATH) == 2


@pytest.mark.parametrize("blank_value", ["", " ", "\t \n"])
def test_blank_or_whitespace_db_env_rejected(
    blank_value: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(HERMES_FINANCE_DB_PATH, blank_value)

    exit_code = main(ingest_args())

    assert exit_code == 1
    captured = capsys.readouterr()
    assert HERMES_FINANCE_DB_PATH in captured.err
    assert captured.out == ""


# ---------------------------------------------------------------------------
# L: the connection always closes, on success and on failure
# ---------------------------------------------------------------------------


def test_connection_closes_on_success_and_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    use_temp_database(monkeypatch, tmp_path)
    opened: list[sqlite3.Connection] = []
    real_open = cli_module._open_finance_database

    def recording_open(database_path: Path) -> sqlite3.Connection:
        connection = real_open(database_path)
        opened.append(connection)
        return connection

    monkeypatch.setattr(cli_module, "_open_finance_database", recording_open)

    assert main(["month", "--year", "2026", "--month", "5"]) == 0
    capsys.readouterr()
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute("SELECT 1")

    with pytest.raises(TransactionParseError):
        main(ingest_args("текст без знака и суммы"))
    assert len(opened) == 2
    with pytest.raises(sqlite3.ProgrammingError):
        opened[1].execute("SELECT 1")


# ---------------------------------------------------------------------------
# only two commands, nothing else
# ---------------------------------------------------------------------------


def test_no_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2


@pytest.mark.parametrize(
    "unknown_command", ["recent", "edit", "delete", "serve", "ingest", "report"]
)
def test_unknown_command_rejected(unknown_command: str) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([unknown_command])
    assert exit_info.value.code == 2


# ---------------------------------------------------------------------------
# M: no Hermes dependency/import/path, no wall clock
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    [
        "hermes_agent",
        "hermes_cli",
        "gateway",
        ".hermes",
        "import telegram",
        "from telegram",
        "python-telegram-bot",
        "aiogram",
        "requests",
        "urllib",
        "socket",
        "http",
        "utcnow",
        "date.today",
        "time.time",
        "time.sleep",
        "perf_counter",
        "monotonic",
    ],
)
def test_cli_source_avoids_forbidden_references(forbidden: str) -> None:
    assert forbidden not in CLI_SOURCE


def test_cli_clock_access_is_exactly_one_aware_utc_snapshot() -> None:
    """The single clock seam is the only wall-clock access in the CLI.

    The seam captures one aware UTC instant: there is no naive
    ``datetime.now()`` call (which would depend on the server-local
    timezone), no no-arg ``astimezone()`` conversion (which would
    convert to the server-local timezone), and exactly one
    ``datetime.now(UTC)`` call site in the whole module.
    """
    assert "datetime.now()" not in CLI_SOURCE
    assert "astimezone()" not in CLI_SOURCE
    assert CLI_SOURCE.count("datetime.now(UTC)") == 1
    assert callable(cli_module._capture_reference_time)


def test_cli_source_has_no_default_or_fallback_thread_id() -> None:
    """No fabricated thread provenance exists in production CLI source.

    ``--thread-id`` is required at the argparse level, so no default
    value, no fallback constant, and no inferred thread id can exist.
    """
    assert "DEFAULT_THREAD_ID" not in CLI_SOURCE
    assert "default=" not in CLI_SOURCE
    assert cli_module.__dict__.get("DEFAULT_THREAD_ID") is None


def test_cli_never_imports_hermes_host_modules() -> None:
    for module_name in sys.modules:
        assert not module_name.startswith("hermes_agent")
        assert not module_name.startswith("hermes_cli")


def test_cli_module_public_api_is_small_and_deliberate() -> None:
    assert sorted(cli_module.__all__) == [
        "FinanceCliError",
        "HERMES_FINANCE_DB_PATH",
        "main",
    ]


# ---------------------------------------------------------------------------
# N: `transactions` command family (stage G1)
# ---------------------------------------------------------------------------


def ingest_many(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    count: int,
) -> None:
    """Ingest ``count`` transactions on distinct September 2026 dates."""
    use_temp_database(monkeypatch, tmp_path)
    for index in range(count):
        assert (
            main(
                ingest_args(
                    "+1 Работа Проект A",
                    message_id=50 + index,
                    update_id=1000 + index,
                    transaction_date=f"2026-09-{index + 1:02d}",
                )
            )
            == 0
        )


def test_transactions_recent_defaults_to_ten(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ingest_many(monkeypatch, tmp_path, count=12)
    capsys.readouterr()

    exit_code = main(["transactions", "recent"])

    assert exit_code == 0
    out = capsys.readouterr().out
    card_lines = [line for line in out.split("\n") if line.startswith("🟢")]
    assert len(card_lines) == 10
    assert out.split("\n")[0] == "🧾 FINANCE | ПОСЛЕДНИЕ ОПЕРАЦИИ"
    # The ten newest (2026-09-12 down to 2026-09-03), newest first.
    assert "12.09.2026" in card_lines[0]
    assert "03.09.2026" in card_lines[-1]
    assert "01.09.2026" not in out
    assert not out.endswith("\n")


def test_transactions_recent_explicit_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ingest_many(monkeypatch, tmp_path, count=12)
    capsys.readouterr()

    exit_code = main(["transactions", "recent", "--limit", "5"])

    assert exit_code == 0
    out = capsys.readouterr().out
    card_lines = [line for line in out.split("\n") if line.startswith("🟢")]
    assert len(card_lines) == 5
    assert "12.09.2026" in card_lines[0]
    assert "08.09.2026" in card_lines[-1]


@pytest.mark.parametrize("bad_limit", ["0", "-3", "101"])
def test_transactions_recent_invalid_limit_propagates_accepted_validation(
    bad_limit: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Out-of-range limits fail through the accepted D1 validation."""
    use_temp_database(monkeypatch, tmp_path)

    with pytest.raises(ValueError):
        main(["transactions", "recent", "--limit", bad_limit])


def test_transactions_recent_non_integer_limit_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_temp_database(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exit_info:
        main(["transactions", "recent", "--limit", "five"])
    assert exit_info.value.code == 2


def test_transactions_recent_empty_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    use_temp_database(monkeypatch, tmp_path)

    exit_code = main(["transactions", "recent"])

    assert exit_code == 0
    assert capsys.readouterr().out == (
        "🧾 FINANCE | ПОСЛЕДНИЕ ОПЕРАЦИИ\n\n📭 Операций нет."
    )


def test_transactions_date_exact_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    use_temp_database(monkeypatch, tmp_path)
    assert (
        main(
            ingest_args(
                "+25 Работа Проект A",
                message_id=50,
                update_id=1000,
                transaction_date="2026-09-05",
            )
        )
        == 0
    )
    assert (
        main(
            ingest_args(
                "-10 Инфраструктура Сервер B | продление сервера",
                message_id=51,
                update_id=1001,
                transaction_date="2026-09-05",
            )
        )
        == 0
    )
    assert (
        main(
            ingest_args(
                "+5 Сервисы Подписка",
                message_id=52,
                update_id=1002,
                transaction_date="2026-09-04",
            )
        )
        == 0
    )
    capsys.readouterr()

    exit_code = main(["transactions", "date", "--date", "2026-09-05"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert out.split("\n")[0] == "🧾 FINANCE | ОПЕРАЦИИ 05.09.2026"
    card_lines = [line for line in out.split("\n") if line.startswith(("🟢", "🔴"))]
    # Same date: id DESC puts the later-persisted expense first.
    assert [line.split(" | ")[0] for line in card_lines] == ["🔴 #2", "🟢 #1"]
    assert "04.09.2026" not in out
    assert not out.endswith("\n")


def test_transactions_date_empty_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    use_temp_database(monkeypatch, tmp_path)

    exit_code = main(["transactions", "date", "--date", "2026-09-05"])

    assert exit_code == 0
    assert capsys.readouterr().out == (
        "🧾 FINANCE | ОПЕРАЦИИ 05.09.2026\n\n📭 Операций нет."
    )


@pytest.mark.parametrize(
    "bad_date",
    ["2026-9-5", "05.09.2026", "20260905", "2026-02-30", "not-a-date", ""],
)
def test_transactions_date_invalid_value_rejected_before_database_work(
    bad_date: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)

    exit_code = main(["transactions", "date", "--date", bad_date])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "--date" in captured.err
    assert captured.out == ""
    assert not database_path.exists()


def test_transactions_date_missing_period_is_a_cli_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Neither ``--date`` nor ``--relative`` is a deterministic CLI error."""
    database_path = use_temp_database(monkeypatch, tmp_path)

    exit_code = main(["transactions", "date"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "either --date or --relative is required" in captured.err
    assert captured.out == ""
    assert not database_path.exists()


def test_transactions_month_exact_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    use_temp_database(monkeypatch, tmp_path)
    assert (
        main(
            ingest_args(
                "+25 Работа Проект A",
                message_id=50,
                update_id=1000,
                transaction_date="2026-09-05",
            )
        )
        == 0
    )
    assert (
        main(
            ingest_args(
                "+5 Сервисы Подписка",
                message_id=51,
                update_id=1001,
                transaction_date="2026-08-20",
            )
        )
        == 0
    )
    capsys.readouterr()

    exit_code = main(["transactions", "month", "--year", "2026", "--month", "9"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert out.split("\n")[0] == "🧾 FINANCE | ОПЕРАЦИИ СЕНТЯБРЬ 2026"
    assert "🟢 #1 | +25 USDT | 05.09.2026" in out
    assert "20.08.2026" not in out
    assert not out.endswith("\n")


def test_transactions_month_empty_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    use_temp_database(monkeypatch, tmp_path)

    exit_code = main(["transactions", "month", "--year", "2026", "--month", "9"])

    assert exit_code == 0
    assert capsys.readouterr().out == (
        "🧾 FINANCE | ОПЕРАЦИИ СЕНТЯБРЬ 2026\n\n📭 Операций нет."
    )


def test_transactions_month_invalid_month_propagates_accepted_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_temp_database(monkeypatch, tmp_path)

    with pytest.raises(ValueError):
        main(["transactions", "month", "--year", "2026", "--month", "13"])


def test_transactions_month_requires_subcommand(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_temp_database(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exit_info:
        main(["transactions"])
    assert exit_info.value.code == 2


def test_transactions_and_month_report_stay_distinct(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`month` is the aggregated report; `transactions month` is the list."""
    use_temp_database(monkeypatch, tmp_path)
    assert (
        main(
            ingest_args(
                "+25 Работа Проект A",
                message_id=50,
                update_id=1000,
                transaction_date="2026-09-05",
            )
        )
        == 0
    )
    capsys.readouterr()

    assert main(["transactions", "month", "--year", "2026", "--month", "9"]) == 0
    list_out = capsys.readouterr().out
    assert main(["month", "--year", "2026", "--month", "9"]) == 0
    report_out = capsys.readouterr().out

    assert list_out.split("\n")[0] == "🧾 FINANCE | ОПЕРАЦИИ СЕНТЯБРЬ 2026"
    assert "🟢 #1 | +25 USDT | 05.09.2026" in list_out
    assert "Доход:" not in list_out
    assert report_out.split("\n")[0] == "💰 FINANCE | СЕНТЯБРЬ 2026"
    assert "📈 Доход: 25 USDT" in report_out


def test_missing_db_env_rejected_for_transactions_commands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(HERMES_FINANCE_DB_PATH, raising=False)

    assert main(["transactions", "recent"]) == 1
    assert main(["transactions", "date", "--date", "2026-09-05"]) == 1
    assert main(["transactions", "month", "--year", "2026", "--month", "9"]) == 1

    errors = capsys.readouterr()
    assert errors.out == ""
    assert errors.err.count(HERMES_FINANCE_DB_PATH) == 3


def test_transactions_connection_closes_on_success_and_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    use_temp_database(monkeypatch, tmp_path)
    opened: list[sqlite3.Connection] = []
    real_open = cli_module._open_finance_database

    def recording_open(database_path: Path) -> sqlite3.Connection:
        connection = real_open(database_path)
        opened.append(connection)
        return connection

    monkeypatch.setattr(cli_module, "_open_finance_database", recording_open)

    assert main(["transactions", "recent"]) == 0
    capsys.readouterr()
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute("SELECT 1")

    # An invalid accepted-layer input (out-of-range limit) still closes.
    with pytest.raises(ValueError):
        main(["transactions", "recent", "--limit", "0"])
    assert len(opened) == 2
    with pytest.raises(sqlite3.ProgrammingError):
        opened[1].execute("SELECT 1")


# ---------------------------------------------------------------------------
# O: `summary month` command family (stage G2)
# ---------------------------------------------------------------------------


def ingest_september_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ingest the canonical seven-transaction Работа month (2026-09)."""
    use_temp_database(monkeypatch, tmp_path)
    rows = [
        ("+50 Работа Проект A", 50),
        ("+30 Работа Проект A", 51),
        ("-6 Работа Проект A", 52),
        ("-4 Работа Проект A", 53),
        ("+25 Работа Проект B", 54),
        ("+20 Работа Проект B", 55),
        ("-10 Работа Проект B", 56),
    ]
    for text, message_id in rows:
        assert (
            main(
                ingest_args(
                    text,
                    message_id=message_id,
                    update_id=1000 + message_id,
                    transaction_date="2026-09-05",
                )
            )
            == 0
        )


def test_summary_month_category_exact_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ingest_september_work(monkeypatch, tmp_path)
    capsys.readouterr()

    exit_code = main(
        ["summary", "month", "--year", "2026", "--month", "9", "--category", "Работа"]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert out == (
        "💰 FINANCE | СЕНТЯБРЬ 2026\n"
        "\n"
        "🔎 Категория: Работа\n"
        "\n"
        "📈 Доход: 125 USDT\n"
        "📉 Расход: 20 USDT\n"
        "⚖️ Итог: +105 USDT\n"
        "🧾 Операций: 7"
    )
    assert not out.endswith("\n")


def test_summary_month_category_source_exact_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ingest_september_work(monkeypatch, tmp_path)
    capsys.readouterr()

    exit_code = main(
        [
            "summary", "month",
            "--year", "2026",
            "--month", "9",
            "--category", "Работа",
            "--source", "Проект A",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert out == (
        "💰 FINANCE | СЕНТЯБРЬ 2026\n"
        "\n"
        "🔎 Работа • Проект A\n"
        "\n"
        "📈 Доход: 80 USDT\n"
        "📉 Расход: 10 USDT\n"
        "⚖️ Итог: +70 USDT\n"
        "🧾 Операций: 4"
    )
    assert not out.endswith("\n")


def test_summary_month_missing_category_renders_no_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ingest_september_work(monkeypatch, tmp_path)
    capsys.readouterr()

    exit_code = main(
        ["summary", "month", "--year", "2026", "--month", "9", "--category", "Инфраструктура"]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == (
        "💰 FINANCE | СЕНТЯБРЬ 2026\n"
        "\n"
        "🔎 Категория: Инфраструктура\n"
        "\n"
        "📭 Данных нет."
    )


def test_summary_month_missing_source_renders_no_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ingest_september_work(monkeypatch, tmp_path)
    capsys.readouterr()

    exit_code = main(
        [
            "summary", "month",
            "--year", "2026",
            "--month", "9",
            "--category", "Работа",
            "--source", "Сервер B",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == (
        "💰 FINANCE | СЕНТЯБРЬ 2026\n"
        "\n"
        "🔎 Работа • Сервер B\n"
        "\n"
        "📭 Данных нет."
    )


def test_summary_month_category_required(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_temp_database(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exit_info:
        main(["summary", "month", "--year", "2026", "--month", "9"])
    assert exit_info.value.code == 2


def test_summary_month_source_without_category_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A source without a category is structurally impossible."""
    use_temp_database(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exit_info:
        main(
            ["summary", "month", "--year", "2026", "--month", "9", "--source", "Проект A"]
        )
    assert exit_info.value.code == 2


def test_summary_month_requires_subcommand(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_temp_database(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exit_info:
        main(["summary"])
    assert exit_info.value.code == 2


def test_summary_month_invalid_month_propagates_accepted_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_temp_database(monkeypatch, tmp_path)

    with pytest.raises(ValueError):
        main(
            [
                "summary", "month",
                "--year", "2026",
                "--month", "13",
                "--category", "Работа",
            ]
        )


def test_summary_month_label_whitespace_handed_to_accepted_normalisation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ingest_september_work(monkeypatch, tmp_path)
    capsys.readouterr()

    exit_code = main(
        ["summary", "month", "--year", "2026", "--month", "9", "--category", " Работа "]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert out.split("\n")[2] == "🔎 Категория: Работа"
    assert "📈 Доход: 125 USDT" in out


def test_summary_month_missing_db_env_rejected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(HERMES_FINANCE_DB_PATH, raising=False)

    assert (
        main(
            ["summary", "month", "--year", "2026", "--month", "9", "--category", "Работа"]
        )
        == 1
    )
    assert (
        main(
            [
                "summary", "month",
                "--year", "2026",
                "--month", "9",
                "--category", "Работа",
                "--source", "Проект A",
            ]
        )
        == 1
    )

    errors = capsys.readouterr()
    assert errors.out == ""
    assert errors.err.count(HERMES_FINANCE_DB_PATH) == 2


def test_summary_month_connection_closes_on_success_and_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    use_temp_database(monkeypatch, tmp_path)
    opened: list[sqlite3.Connection] = []
    real_open = cli_module._open_finance_database

    def recording_open(database_path: Path) -> sqlite3.Connection:
        connection = real_open(database_path)
        opened.append(connection)
        return connection

    monkeypatch.setattr(cli_module, "_open_finance_database", recording_open)

    assert (
        main(
            ["summary", "month", "--year", "2026", "--month", "9", "--category", "Работа"]
        )
        == 0
    )
    capsys.readouterr()
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute("SELECT 1")

    # An invalid accepted-layer input (out-of-range month) still closes.
    with pytest.raises(ValueError):
        main(
            [
                "summary", "month",
                "--year", "2026",
                "--month", "13",
                "--category", "Работа",
            ]
        )
    assert len(opened) == 2
    with pytest.raises(sqlite3.ProgrammingError):
        opened[1].execute("SELECT 1")


def test_summary_month_stays_distinct_from_month_and_transactions_month(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Three September views: aggregate, filtered aggregate, list."""
    ingest_september_work(monkeypatch, tmp_path)
    capsys.readouterr()

    assert (
        main(["summary", "month", "--year", "2026", "--month", "9", "--category", "Работа"])
        == 0
    )
    summary_out = capsys.readouterr().out
    assert main(["month", "--year", "2026", "--month", "9"]) == 0
    report_out = capsys.readouterr().out
    assert main(["transactions", "month", "--year", "2026", "--month", "9"]) == 0
    list_out = capsys.readouterr().out

    assert summary_out.split("\n")[0] == "💰 FINANCE | СЕНТЯБРЬ 2026"
    assert summary_out.split("\n")[2] == "🔎 Категория: Работа"
    assert "📂" not in summary_out
    assert "📂 Работа" in report_out
    assert "🟢 #1 | +50 USDT" in list_out
    assert "🟢" not in summary_out


# ---------------------------------------------------------------------------
# P: relative periods (stage G3)
# ---------------------------------------------------------------------------

#: Fixed aware instant used as the patched clock snapshot: 12:00 UTC on
#: 2026-09-05, which is 15:00 MSK on the same business date.
CLOCK_SEPTEMBER_2026: Final[datetime] = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)

#: 21:30 UTC on 2026-09-05: still September 5 in UTC and New York, but
#: already September 6 in Moscow.
CLOCK_NEAR_MOSCOW_MIDNIGHT: Final[datetime] = datetime(2026, 9, 5, 21, 30, tzinfo=UTC)

#: The three month-style commands, with their stable argument prefixes.
MONTH_COMMAND_ARGV: Final[dict[str, list[str]]] = {
    "month": ["month"],
    "transactions-month": ["transactions", "month"],
    "summary-month": ["summary", "month", "--category", "Работа"],
}


def patch_clock(
    monkeypatch: pytest.MonkeyPatch, instant: datetime
) -> list[datetime]:
    """Patch the CLI clock seam with one fixed aware instant.

    Returns the list the seam appends every captured snapshot to, so
    tests can also assert how often the clock was read.
    """
    captured: list[datetime] = []

    def fixed_clock() -> datetime:
        captured.append(instant)
        return instant

    monkeypatch.setattr(cli_module, "_capture_reference_time", fixed_clock)
    return captured


def test_transactions_date_relative_today(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bounded_tz_database: None,
) -> None:
    use_temp_database(monkeypatch, tmp_path)
    assert (
        main(
            ingest_args(
                "+25 Работа Проект A",
                message_id=50,
                update_id=1000,
                transaction_date="2026-09-05",
            )
        )
        == 0
    )
    assert (
        main(
            ingest_args(
                "+5 Сервисы Подписка",
                message_id=51,
                update_id=1001,
                transaction_date="2026-09-04",
            )
        )
        == 0
    )
    capsys.readouterr()
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    exit_code = main(
        ["transactions", "date", "--relative", "today", "--timezone", "Europe/Moscow"]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    # The output title carries the resolved business date, never vague
    # relative wording.
    assert out.split("\n")[0] == "🧾 FINANCE | ОПЕРАЦИИ 05.09.2026"
    assert "🟢 #1 | +25 USDT | 05.09.2026" in out
    assert "04.09.2026" not in out
    assert "СЕГОДНЯ" not in out


def test_transactions_date_relative_yesterday(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bounded_tz_database: None,
) -> None:
    use_temp_database(monkeypatch, tmp_path)
    assert (
        main(
            ingest_args(
                "+25 Работа Проект A",
                message_id=50,
                update_id=1000,
                transaction_date="2026-09-05",
            )
        )
        == 0
    )
    assert (
        main(
            ingest_args(
                "+5 Сервисы Подписка",
                message_id=51,
                update_id=1001,
                transaction_date="2026-09-06",
            )
        )
        == 0
    )
    capsys.readouterr()
    patch_clock(monkeypatch, datetime(2026, 9, 6, 12, 0, tzinfo=UTC))

    exit_code = main(
        [
            "transactions",
            "date",
            "--relative",
            "yesterday",
            "--timezone",
            "Europe/Moscow",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert out.split("\n")[0] == "🧾 FINANCE | ОПЕРАЦИИ 05.09.2026"
    assert "🟢 #1 | +25 USDT | 05.09.2026" in out
    assert "06.09.2026" not in out


def test_transactions_date_relative_near_midnight_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bounded_tz_database: None,
) -> None:
    """21:30 UTC is the next Moscow business date but not a New York one."""
    use_temp_database(monkeypatch, tmp_path)
    assert (
        main(
            ingest_args(
                "+25 Работа Проект A",
                message_id=50,
                update_id=1000,
                transaction_date="2026-09-06",
            )
        )
        == 0
    )
    capsys.readouterr()
    patch_clock(monkeypatch, CLOCK_NEAR_MOSCOW_MIDNIGHT)

    assert (
        main(
            ["transactions", "date", "--relative", "today", "--timezone", "Europe/Moscow"]
        )
        == 0
    )
    moscow_out = capsys.readouterr().out
    assert moscow_out.split("\n")[0] == "🧾 FINANCE | ОПЕРАЦИИ 06.09.2026"

    assert (
        main(
            [
                "transactions",
                "date",
                "--relative",
                "today",
                "--timezone",
                "America/New_York",
            ]
        )
        == 0
    )
    new_york_out = capsys.readouterr().out
    assert new_york_out.split("\n")[0] == "🧾 FINANCE | ОПЕРАЦИИ 05.09.2026"
    assert "📭 Операций нет." in new_york_out


def test_transactions_date_relative_yesterday_from_near_midnight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bounded_tz_database: None,
) -> None:
    use_temp_database(monkeypatch, tmp_path)
    patch_clock(monkeypatch, CLOCK_NEAR_MOSCOW_MIDNIGHT)

    exit_code = main(
        [
            "transactions",
            "date",
            "--relative",
            "yesterday",
            "--timezone",
            "Europe/Moscow",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert out.split("\n")[0] == "🧾 FINANCE | ОПЕРАЦИИ 05.09.2026"
    assert "📭 Операций нет." in out


def test_transactions_date_explicit_and_relative_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bounded_tz_database: None,
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    exit_code = main(
        [
            "transactions",
            "date",
            "--date",
            "2026-09-05",
            "--relative",
            "today",
            "--timezone",
            "Europe/Moscow",
        ]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "mutually exclusive" in captured.err
    assert captured.out == ""
    assert not database_path.exists()


def test_transactions_date_relative_without_timezone_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bounded_tz_database: None,
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    exit_code = main(["transactions", "date", "--relative", "today"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "--timezone" in captured.err
    assert "no default timezone is assumed" in captured.err
    assert captured.out == ""
    assert not database_path.exists()


@pytest.mark.parametrize(
    "bad_timezone",
    ["Europe/Krasnoy", "Not/AZone", "utc", "Moscow", "Europe//Moscow", "", "   "],
)
def test_transactions_date_invalid_timezone_rejected_before_database_work(
    bad_timezone: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bounded_tz_database: None,
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    exit_code = main(
        ["transactions", "date", "--relative", "today", "--timezone", bad_timezone]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "--timezone" in captured.err
    assert captured.out == ""
    assert not database_path.exists()


@pytest.mark.parametrize("bad_token", ["tomorrow", "current", "TODAY", "yesterdays"])
def test_transactions_date_unsupported_relative_token_is_a_usage_error(
    bad_token: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bounded_tz_database: None,
) -> None:
    use_temp_database(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "transactions",
                "date",
                "--relative",
                bad_token,
                "--timezone",
                "Europe/Moscow",
            ]
        )
    assert exit_info.value.code == 2


def test_transactions_month_relative_current(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bounded_tz_database: None,
) -> None:
    use_temp_database(monkeypatch, tmp_path)
    assert (
        main(
            ingest_args(
                "+25 Работа Проект A",
                message_id=50,
                update_id=1000,
                transaction_date="2026-09-05",
            )
        )
        == 0
    )
    assert (
        main(
            ingest_args(
                "+5 Сервисы Подписка",
                message_id=51,
                update_id=1001,
                transaction_date="2026-08-20",
            )
        )
        == 0
    )
    capsys.readouterr()
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    exit_code = main(
        [
            "transactions",
            "month",
            "--relative",
            "current",
            "--timezone",
            "Europe/Moscow",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert out.split("\n")[0] == "🧾 FINANCE | ОПЕРАЦИИ СЕНТЯБРЬ 2026"
    assert "🟢 #1 | +25 USDT | 05.09.2026" in out
    assert "20.08.2026" not in out
    assert "ТЕКУЩИЙ" not in out


def test_transactions_month_relative_previous(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bounded_tz_database: None,
) -> None:
    use_temp_database(monkeypatch, tmp_path)
    assert (
        main(
            ingest_args(
                "+25 Работа Проект A",
                message_id=50,
                update_id=1000,
                transaction_date="2026-09-05",
            )
        )
        == 0
    )
    assert (
        main(
            ingest_args(
                "+5 Сервисы Подписка",
                message_id=51,
                update_id=1001,
                transaction_date="2026-08-20",
            )
        )
        == 0
    )
    capsys.readouterr()
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    exit_code = main(
        [
            "transactions",
            "month",
            "--relative",
            "previous",
            "--timezone",
            "Europe/Moscow",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert out.split("\n")[0] == "🧾 FINANCE | ОПЕРАЦИИ АВГУСТ 2026"
    assert "🟢 #2 | +5 USDT | 20.08.2026" in out
    assert "05.09.2026" not in out


def test_month_relative_current(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bounded_tz_database: None,
) -> None:
    use_temp_database(monkeypatch, tmp_path)
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    exit_code = main(["month", "--relative", "current", "--timezone", "Europe/Moscow"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert out.split("\n")[0] == "💰 FINANCE | СЕНТЯБРЬ 2026"
    assert "📭 Операций за месяц нет." in out


def test_month_relative_previous_returns_exact_accepted_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bounded_tz_database: None,
) -> None:
    """September clock + previous -> the exact accepted August report."""
    use_temp_database(monkeypatch, tmp_path)
    canonical_messages = [
        "+25 Работа Проект A",
        "+30 Работа Проект B",
        "+5 Сервисы Подписка",
        "-10 Инфраструктура Хостинг",
        "-1 Инфраструктура Сервер A",
        "-10 Инфраструктура Сервер B",
    ]
    for message_id, text in enumerate(canonical_messages, start=50):
        assert (
            main(
                ingest_args(
                    text,
                    message_id=message_id,
                    update_id=1000 + message_id,
                    transaction_date="2026-08-15",
                )
            )
            == 0
        )
    capsys.readouterr()
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    exit_code = main(["month", "--relative", "previous", "--timezone", "Europe/Moscow"])

    assert exit_code == 0
    assert capsys.readouterr().out == CANONICAL_AUGUST_REPORT


def test_month_relative_previous_january_rolls_to_december(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bounded_tz_database: None,
) -> None:
    use_temp_database(monkeypatch, tmp_path)
    patch_clock(monkeypatch, datetime(2027, 1, 10, 12, 0, tzinfo=UTC))

    exit_code = main(["month", "--relative", "previous", "--timezone", "Europe/Moscow"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert out.split("\n")[0] == "💰 FINANCE | ДЕКАБРЬ 2026"
    assert "📭 Операций за месяц нет." in out


def test_summary_month_relative_current_matches_explicit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bounded_tz_database: None,
) -> None:
    ingest_september_work(monkeypatch, tmp_path)
    capsys.readouterr()
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    assert (
        main(
            [
                "summary",
                "month",
                "--relative",
                "current",
                "--timezone",
                "Europe/Moscow",
                "--category",
                "Работа",
            ]
        )
        == 0
    )
    relative_out = capsys.readouterr().out
    assert (
        main(
            ["summary", "month", "--year", "2026", "--month", "9", "--category", "Работа"]
        )
        == 0
    )
    explicit_out = capsys.readouterr().out

    assert relative_out == explicit_out
    assert relative_out.split("\n")[0] == "💰 FINANCE | СЕНТЯБРЬ 2026"
    assert "📈 Доход: 125 USDT" in relative_out


def test_summary_month_relative_previous_matches_explicit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bounded_tz_database: None,
) -> None:
    ingest_september_work(monkeypatch, tmp_path)
    capsys.readouterr()
    # An October clock snapshot resolves `previous` to September 2026.
    patch_clock(monkeypatch, datetime(2026, 10, 2, 12, 0, tzinfo=UTC))

    assert (
        main(
            [
                "summary",
                "month",
                "--relative",
                "previous",
                "--timezone",
                "Europe/Moscow",
                "--category",
                "Работа",
            ]
        )
        == 0
    )
    relative_out = capsys.readouterr().out
    assert (
        main(
            ["summary", "month", "--year", "2026", "--month", "9", "--category", "Работа"]
        )
        == 0
    )
    explicit_out = capsys.readouterr().out

    assert relative_out == explicit_out


def test_summary_month_relative_with_source_keeps_g2_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bounded_tz_database: None,
) -> None:
    ingest_september_work(monkeypatch, tmp_path)
    capsys.readouterr()
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    exit_code = main(
        [
            "summary",
            "month",
            "--relative",
            "current",
            "--timezone",
            "Europe/Moscow",
            "--category",
            "Работа",
            "--source",
            "Проект A",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert out.split("\n")[2] == "🔎 Работа • Проект A"
    assert "📈 Доход: 80 USDT" in out
    assert "🧾 Операций: 4" in out


@pytest.mark.parametrize("command", sorted(MONTH_COMMAND_ARGV))
def test_month_commands_explicit_and_relative_rejected(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bounded_tz_database: None,
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    exit_code = main(
        MONTH_COMMAND_ARGV[command]
        + [
            "--year", "2026",
            "--month", "9",
            "--relative", "current",
            "--timezone", "Europe/Moscow",
        ]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "mutually exclusive" in captured.err
    assert captured.out == ""
    assert not database_path.exists()


@pytest.mark.parametrize("command", sorted(MONTH_COMMAND_ARGV))
@pytest.mark.parametrize("partial", [["--year", "2026"], ["--month", "9"]])
def test_month_commands_partial_explicit_pair_rejected(
    command: str,
    partial: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)

    exit_code = main(MONTH_COMMAND_ARGV[command] + partial)

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "--year and --month must be supplied together" in captured.err
    assert captured.out == ""
    assert not database_path.exists()


@pytest.mark.parametrize("command", sorted(MONTH_COMMAND_ARGV))
def test_month_commands_missing_period_rejected(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)

    exit_code = main(MONTH_COMMAND_ARGV[command])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "either --year and --month together or --relative is required" in captured.err
    assert captured.out == ""
    assert not database_path.exists()


@pytest.mark.parametrize("command", sorted(MONTH_COMMAND_ARGV))
def test_month_commands_relative_without_timezone_rejected(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bounded_tz_database: None,
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    exit_code = main(MONTH_COMMAND_ARGV[command] + ["--relative", "current"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "--timezone" in captured.err
    assert captured.out == ""
    assert not database_path.exists()


@pytest.mark.parametrize("bad_timezone", ["Europe/Krasnoy", "Not/AZone", "", "  "])
@pytest.mark.parametrize("command", sorted(MONTH_COMMAND_ARGV))
def test_month_commands_invalid_timezone_rejected_before_database_work(
    command: str,
    bad_timezone: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bounded_tz_database: None,
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    exit_code = main(
        MONTH_COMMAND_ARGV[command]
        + ["--relative", "current", "--timezone", bad_timezone]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "--timezone" in captured.err
    assert captured.out == ""
    assert not database_path.exists()


@pytest.mark.parametrize("bad_token", ["next", "today", "CURRENT", "last"])
@pytest.mark.parametrize("command", sorted(MONTH_COMMAND_ARGV))
def test_month_commands_unsupported_relative_token_is_a_usage_error(
    command: str,
    bad_token: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bounded_tz_database: None,
) -> None:
    use_temp_database(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exit_info:
        main(
            MONTH_COMMAND_ARGV[command]
            + ["--relative", bad_token, "--timezone", "Europe/Moscow"]
        )
    assert exit_info.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["transactions", "date", "--relative", "today", "--timezone", "Europe/Moscow"],
        [
            "transactions",
            "date",
            "--relative",
            "yesterday",
            "--timezone",
            "Europe/Moscow",
        ],
        ["transactions", "month", "--relative", "current", "--timezone", "Europe/Moscow"],
        ["transactions", "month", "--relative", "previous", "--timezone", "Europe/Moscow"],
        ["month", "--relative", "current", "--timezone", "Europe/Moscow"],
        ["month", "--relative", "previous", "--timezone", "Europe/Moscow"],
        [
            "summary",
            "month",
            "--relative",
            "current",
            "--timezone",
            "Europe/Moscow",
            "--category",
            "Работа",
        ],
        [
            "summary",
            "month",
            "--relative",
            "previous",
            "--timezone",
            "Europe/Moscow",
            "--category",
            "Работа",
        ],
    ],
)
def test_relative_invocation_captures_the_clock_exactly_once(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bounded_tz_database: None,
) -> None:
    use_temp_database(monkeypatch, tmp_path)
    captured = patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    exit_code = main(argv)

    assert exit_code == 0
    assert len(captured) == 1


def test_explicit_commands_never_capture_the_clock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Explicit-period, recent, and ingest paths read no clock at all."""

    def forbidden_clock() -> datetime:
        raise AssertionError("this command must not read the clock")

    monkeypatch.setattr(cli_module, "_capture_reference_time", forbidden_clock)

    use_temp_database(monkeypatch, tmp_path)
    assert main(["month", "--year", "2026", "--month", "9"]) == 0
    assert main(["transactions", "date", "--date", "2026-09-05"]) == 0
    assert main(["transactions", "month", "--year", "2026", "--month", "9"]) == 0
    assert (
        main(["summary", "month", "--year", "2026", "--month", "9", "--category", "Работа"])
        == 0
    )
    assert main(["transactions", "recent"]) == 0
    assert main(ingest_args()) == 0
    capsys.readouterr()


def test_relative_commands_require_db_env(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    bounded_tz_database: None,
) -> None:
    monkeypatch.delenv(HERMES_FINANCE_DB_PATH, raising=False)
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    assert (
        main(["transactions", "date", "--relative", "today", "--timezone", "Europe/Moscow"])
        == 1
    )
    assert (
        main(["transactions", "month", "--relative", "current", "--timezone", "Europe/Moscow"])
        == 1
    )
    assert main(["month", "--relative", "current", "--timezone", "Europe/Moscow"]) == 1
    assert (
        main(
            [
                "summary",
                "month",
                "--relative",
                "current",
                "--timezone",
                "Europe/Moscow",
                "--category",
                "Работа",
            ]
        )
        == 1
    )

    errors = capsys.readouterr()
    assert errors.out == ""
    assert errors.err.count(HERMES_FINANCE_DB_PATH) == 4


def test_relative_connection_closes_on_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bounded_tz_database: None,
) -> None:
    use_temp_database(monkeypatch, tmp_path)
    opened: list[sqlite3.Connection] = []
    real_open = cli_module._open_finance_database

    def recording_open(database_path: Path) -> sqlite3.Connection:
        connection = real_open(database_path)
        opened.append(connection)
        return connection

    monkeypatch.setattr(cli_module, "_open_finance_database", recording_open)
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    assert main(["month", "--relative", "current", "--timezone", "Europe/Moscow"]) == 0
    capsys.readouterr()
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute("SELECT 1")


# ---------------------------------------------------------------------------
# Q: `transaction` mutation command family (stage H1)
# ---------------------------------------------------------------------------


def test_transaction_edit_amount_exact_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)
    assert (
        main(
            ingest_args(
                "-10 Инфраструктура Сервер B | продление сервера",
                message_id=50,
                update_id=1000,
                transaction_date="2026-09-05",
            )
        )
        == 0
    )
    capsys.readouterr()
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    exit_code = main(
        ["transaction", "edit-amount", "--id", "1", "--amount", "12"]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert json.loads(out) == {"disposition": "UPDATED", "transaction_id": "1"}
    assert out.count("\n") == 1

    connection = open_for_inspection(database_path)
    try:
        assert transaction_rows(connection) == [
            (
                "expense",
                "12",
                "Инфраструктура",
                "Сервер B",
                "продление сервера",
                "2026-09-05",
                RECEIVED_AT_TEXT,
                CLOCK_SEPTEMBER_2026.isoformat(),
                "active",
                None,
                -100,
                7,
                50,
                1000,
            )
        ]
        assert processed_update_rows(connection) == [
            (1000, -100, 7, 50, RECEIVED_AT_TEXT)
        ]
    finally:
        connection.close()


def test_transaction_edit_amount_last_targets_newest_active(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)
    assert (
        main(
            ingest_args(
                "-10 Инфраструктура Сервер B",
                message_id=50,
                update_id=1000,
                transaction_date="2026-09-04",
            )
        )
        == 0
    )
    assert (
        main(
            ingest_args(
                "+25 Работа Проект A",
                message_id=51,
                update_id=1001,
                transaction_date="2026-09-05",
            )
        )
        == 0
    )
    capsys.readouterr()
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    exit_code = main(["transaction", "edit-amount", "--last", "--amount", "12"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "disposition": "UPDATED",
        "transaction_id": "2",
    }

    connection = open_for_inspection(database_path)
    try:
        rows = transaction_rows(connection)
        # The newest ACTIVE transaction (#2, 2026-09-05) is the target:
        # amount 25 -> 12, income direction preserved. The older
        # transaction (#1) is untouched.
        assert rows[0][0] == "expense" and rows[0][1] == "10"
        assert rows[1][0] == "income" and rows[1][1] == "12"
    finally:
        connection.close()


def test_transaction_edit_amount_preserves_income_direction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)
    assert main(ingest_args("+25 Работа Проект A")) == 0
    capsys.readouterr()
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    exit_code = main(
        ["transaction", "edit-amount", "--id", "1", "--amount", "12"]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["transaction_id"] == "1"
    connection = open_for_inspection(database_path)
    try:
        row = transaction_rows(connection)[0]
        assert row[0] == "income"
        assert row[1] == "12"
    finally:
        connection.close()


def test_transaction_edit_amount_preserves_expense_direction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The canonical product example: ``-10 Инфраструктура Сервер B`` -> ``-12``."""
    database_path = use_temp_database(monkeypatch, tmp_path)
    assert main(ingest_args("-10 Инфраструктура Сервер B")) == 0
    capsys.readouterr()
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    exit_code = main(["transaction", "edit-amount", "--last", "--amount", "12"])

    assert exit_code == 0
    connection = open_for_inspection(database_path)
    try:
        row = transaction_rows(connection)[0]
        assert row[0] == "expense"
        assert row[1] == "12"
    finally:
        connection.close()


def test_transaction_edit_amount_deleted_target_rejects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)
    assert main(ingest_args("-10 Инфраструктура Сервер B")) == 0
    capsys.readouterr()
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)
    assert main(["transaction", "delete", "--id", "1"]) == 0
    capsys.readouterr()

    with pytest.raises(TransactionNotActiveError):
        main(["transaction", "edit-amount", "--id", "1", "--amount", "12"])

    # No success JSON, and the DELETED row keeps its original values.
    assert capsys.readouterr().out == ""
    connection = open_for_inspection(database_path)
    try:
        row = transaction_rows(connection)[0]
        assert row[1] == "10"
        assert row[8] == "deleted"
    finally:
        connection.close()


def test_transaction_edit_amount_missing_id_rejects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    use_temp_database(monkeypatch, tmp_path)
    assert main(ingest_args("-10 Инфраструктура Сервер B")) == 0
    capsys.readouterr()

    with pytest.raises(TransactionNotFoundError):
        main(["transaction", "edit-amount", "--id", "15", "--amount", "12"])

    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("bad_id", ["abc", "0", "-1", "01", "1.5"])
def test_transaction_edit_amount_invalid_id_rejects(
    bad_id: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)
    assert main(ingest_args("-10 Инфраструктура Сервер B")) == 0

    with pytest.raises((TypeError, ValueError)):
        main(["transaction", "edit-amount", "--id", bad_id, "--amount", "12"])

    connection = open_for_inspection(database_path)
    try:
        assert transaction_rows(connection)[0][1] == "10"
    finally:
        connection.close()


@pytest.mark.parametrize("bad_amount", ["abc", "0", "-5", "12,5"])
def test_transaction_edit_amount_invalid_amount_rejects(
    bad_amount: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)
    assert main(ingest_args("-10 Инфраструктура Сервер B")) == 0

    with pytest.raises(ValueError):
        main(
            ["transaction", "edit-amount", "--id", "1", "--amount", bad_amount]
        )

    connection = open_for_inspection(database_path)
    try:
        row = transaction_rows(connection)[0]
        # A rejected amount never flips the stored direction or amount.
        assert row[0] == "expense"
        assert row[1] == "10"
    finally:
        connection.close()


def test_transaction_delete_exact_id_json_and_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)
    assert (
        main(
            ingest_args(
                "-10 Инфраструктура Сервер B | продление сервера",
                message_id=50,
                update_id=1000,
                transaction_date="2026-09-05",
            )
        )
        == 0
    )
    capsys.readouterr()
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    exit_code = main(["transaction", "delete", "--id", "1"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert json.loads(out) == {"disposition": "DELETED", "transaction_id": "1"}
    assert out.count("\n") == 1

    connection = open_for_inspection(database_path)
    try:
        assert transaction_rows(connection) == [
            (
                "expense",
                "10",
                "Инфраструктура",
                "Сервер B",
                "продление сервера",
                "2026-09-05",
                RECEIVED_AT_TEXT,
                CLOCK_SEPTEMBER_2026.isoformat(),
                "deleted",
                CLOCK_SEPTEMBER_2026.isoformat(),
                -100,
                7,
                50,
                1000,
            )
        ]
        assert processed_update_rows(connection) == [
            (1000, -100, 7, 50, RECEIVED_AT_TEXT)
        ]
    finally:
        connection.close()


def test_transaction_delete_last_targets_newest_active(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)
    assert (
        main(
            ingest_args(
                "-10 Инфраструктура Сервер B",
                message_id=50,
                update_id=1000,
                transaction_date="2026-09-04",
            )
        )
        == 0
    )
    assert (
        main(
            ingest_args(
                "+25 Работа Проект A",
                message_id=51,
                update_id=1001,
                transaction_date="2026-09-05",
            )
        )
        == 0
    )
    capsys.readouterr()
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    exit_code = main(["transaction", "delete", "--last"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "disposition": "DELETED",
        "transaction_id": "2",
    }
    connection = open_for_inspection(database_path)
    try:
        rows = transaction_rows(connection)
        assert rows[0][8] == "active"
        assert rows[1][8] == "deleted"
    finally:
        connection.close()


def test_transaction_delete_repeated_exact_id_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)
    assert main(ingest_args("-10 Инфраструктура Сервер B")) == 0
    capsys.readouterr()
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)
    assert main(["transaction", "delete", "--id", "1"]) == 0
    capsys.readouterr()

    patch_clock(monkeypatch, datetime(2026, 9, 7, 9, 0, tzinfo=UTC))
    exit_code = main(["transaction", "delete", "--id", "1"])

    # The requested final state is satisfied again; the original
    # deleted_at/updated_at are never rewritten.
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "disposition": "DELETED",
        "transaction_id": "1",
    }
    connection = open_for_inspection(database_path)
    try:
        row = transaction_rows(connection)[0]
        assert row[8] == "deleted"
        assert row[9] == CLOCK_SEPTEMBER_2026.isoformat()
        # No hard delete: the row still exists.
        assert len(transaction_rows(connection)) == 1
    finally:
        connection.close()


def test_transaction_delete_missing_id_rejects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    use_temp_database(monkeypatch, tmp_path)
    assert main(ingest_args("-10 Инфраструктура Сервер B")) == 0

    with pytest.raises(TransactionNotFoundError):
        main(["transaction", "delete", "--id", "15"])


@pytest.mark.parametrize("bad_id", ["abc", "0", "-1", "01", "1.5"])
def test_transaction_delete_invalid_id_rejects(
    bad_id: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)
    assert main(ingest_args("-10 Инфраструктура Сервер B")) == 0

    with pytest.raises((TypeError, ValueError)):
        main(["transaction", "delete", "--id", bad_id])

    connection = open_for_inspection(database_path)
    try:
        assert transaction_rows(connection)[0][8] == "active"
    finally:
        connection.close()


@pytest.mark.parametrize(
    "argv",
    [
        ["transaction", "edit-amount", "--id", "1", "--last", "--amount", "12"],
        ["transaction", "delete", "--id", "1", "--last"],
    ],
)
def test_transaction_id_and_last_are_mutually_exclusive(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)

    exit_code = main(argv)

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "mutually exclusive" in captured.err
    assert captured.out == ""
    assert not database_path.exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["transaction", "edit-amount", "--amount", "12"],
        ["transaction", "delete"],
    ],
)
def test_transaction_target_required(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)

    exit_code = main(argv)

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "either --id or --last is required" in captured.err
    assert captured.out == ""
    assert not database_path.exists()


def test_transaction_edit_amount_amount_required_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_temp_database(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exit_info:
        main(["transaction", "edit-amount", "--id", "1"])
    assert exit_info.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["transaction", "edit-amount", "--last", "--amount", "12"],
        ["transaction", "delete", "--last"],
    ],
)
def test_transaction_last_empty_ledger_fails_with_no_write(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = use_temp_database(monkeypatch, tmp_path)

    exit_code = main(argv)

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "--last requires at least one ACTIVE transaction" in captured.err
    assert captured.out == ""
    connection = open_for_inspection(database_path)
    try:
        assert transaction_rows(connection) == []
        assert processed_update_rows(connection) == []
    finally:
        connection.close()


def test_transaction_last_skips_soft_deleted_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A soft-deleted newest row is not a candidate for ``--last``."""
    database_path = use_temp_database(monkeypatch, tmp_path)
    assert (
        main(
            ingest_args(
                "-10 Инфраструктура Сервер B",
                message_id=50,
                update_id=1000,
                transaction_date="2026-09-04",
            )
        )
        == 0
    )
    assert (
        main(
            ingest_args(
                "+25 Работа Проект A",
                message_id=51,
                update_id=1001,
                transaction_date="2026-09-05",
            )
        )
        == 0
    )
    capsys.readouterr()
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)
    assert main(["transaction", "delete", "--last"]) == 0
    assert json.loads(capsys.readouterr().out)["transaction_id"] == "2"

    # --last now resolves to the remaining ACTIVE transaction (#1).
    exit_code = main(["transaction", "edit-amount", "--last", "--amount", "12"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "disposition": "UPDATED",
        "transaction_id": "1",
    }
    connection = open_for_inspection(database_path)
    try:
        rows = transaction_rows(connection)
        assert rows[0][1] == "12"
        assert rows[0][8] == "active"
        assert rows[1][8] == "deleted"
    finally:
        connection.close()


def test_transaction_last_matches_recent_limit_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--last`` has exactly the ``transactions recent --limit 1`` target."""
    use_temp_database(monkeypatch, tmp_path)
    for index, (text, day) in enumerate(
        (
            ("-10 Инфраструктура Сервер B", "2026-09-01"),
            ("+25 Работа Проект A", "2026-09-02"),
            ("-1 Инфраструктура Сервер A", "2026-09-03"),
        )
    ):
        assert (
            main(
                ingest_args(
                    text,
                    message_id=50 + index,
                    update_id=1000 + index,
                    transaction_date=day,
                )
            )
            == 0
        )
    capsys.readouterr()
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    assert main(["transactions", "recent", "--limit", "1"]) == 0
    recent_out = capsys.readouterr().out
    assert "#3" in recent_out

    exit_code = main(["transaction", "edit-amount", "--last", "--amount", "12"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["transaction_id"] == "3"


def test_transaction_cli_source_adds_no_competing_sql() -> None:
    """The CLI resolves --last only through the accepted D1 operation."""
    assert "list_recent_transactions" in CLI_SOURCE
    for forbidden in ("ORDER BY", "SELECT ", "DELETE FROM", "UPDATE ", "LIMIT ?"):
        assert forbidden not in CLI_SOURCE


@pytest.mark.parametrize(
    "argv",
    [
        ["transaction", "edit-amount", "--id", "1", "--amount", "12"],
        ["transaction", "edit-amount", "--last", "--amount", "12"],
        ["transaction", "delete", "--id", "1"],
        ["transaction", "delete", "--last"],
    ],
)
def test_transaction_mutation_captures_the_clock_exactly_once(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    use_temp_database(monkeypatch, tmp_path)
    assert main(ingest_args("-10 Инфраструктура Сервер B")) == 0
    capsys.readouterr()
    captured = patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    exit_code = main(argv)

    assert exit_code == 0
    assert len(captured) == 1
    assert captured[0].tzinfo is UTC


def test_transaction_mutation_needs_no_business_timezone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mutations need only the database env: no --timezone, no zone config."""
    database_path = use_temp_database(monkeypatch, tmp_path)
    assert main(ingest_args("-10 Инфраструктура Сервер B")) == 0
    capsys.readouterr()

    edit_argv = ["transaction", "edit-amount", "--last", "--amount", "12"]
    delete_argv = ["transaction", "delete", "--id", "1"]
    assert "--timezone" not in edit_argv + delete_argv

    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)
    assert main(edit_argv) == 0
    assert json.loads(capsys.readouterr().out)["disposition"] == "UPDATED"
    assert main(delete_argv) == 0
    assert json.loads(capsys.readouterr().out)["disposition"] == "DELETED"

    connection = open_for_inspection(database_path)
    try:
        row = transaction_rows(connection)[0]
        # One snapshot per invocation: the edit stamp first, then the
        # (identical patched) delete stamp.
        assert row[7] == CLOCK_SEPTEMBER_2026.isoformat()
        assert row[8] == "deleted"
        assert row[9] == CLOCK_SEPTEMBER_2026.isoformat()
    finally:
        connection.close()


def test_transaction_mutation_requires_db_env(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(HERMES_FINANCE_DB_PATH, raising=False)

    assert main(["transaction", "edit-amount", "--id", "1", "--amount", "12"]) == 1
    assert main(["transaction", "delete", "--last"]) == 1

    errors = capsys.readouterr()
    assert errors.out == ""
    assert errors.err.count(HERMES_FINANCE_DB_PATH) == 2


def test_transaction_mutation_connection_closes_on_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    use_temp_database(monkeypatch, tmp_path)
    assert main(ingest_args("-10 Инфраструктура Сервер B")) == 0
    capsys.readouterr()
    opened: list[sqlite3.Connection] = []
    real_open = cli_module._open_finance_database

    def recording_open(database_path: Path) -> sqlite3.Connection:
        connection = real_open(database_path)
        opened.append(connection)
        return connection

    monkeypatch.setattr(cli_module, "_open_finance_database", recording_open)
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    assert main(["transaction", "edit-amount", "--id", "1", "--amount", "12"]) == 0
    capsys.readouterr()
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute("SELECT 1")

    # An accepted-layer failure (missing target) still closes.
    with pytest.raises(TransactionNotFoundError):
        main(["transaction", "edit-amount", "--id", "99", "--amount", "12"])
    assert len(opened) == 2
    with pytest.raises(sqlite3.ProgrammingError):
        opened[1].execute("SELECT 1")


def test_transaction_requires_subcommand(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_temp_database(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exit_info:
        main(["transaction"])
    assert exit_info.value.code == 2


def test_read_choose_mutate_flow_through_the_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The product flow: list with IDs, then mutate by the chosen ID."""
    use_temp_database(monkeypatch, tmp_path)
    assert (
        main(
            ingest_args(
                "-10 Инфраструктура Сервер B",
                message_id=50,
                update_id=1000,
                transaction_date="2026-09-04",
            )
        )
        == 0
    )
    assert (
        main(
            ingest_args(
                "+25 Работа Проект A",
                message_id=51,
                update_id=1001,
                transaction_date="2026-09-05",
            )
        )
        == 0
    )
    capsys.readouterr()
    patch_clock(monkeypatch, CLOCK_SEPTEMBER_2026)

    # Read: the user sees the transaction IDs.
    assert main(["transactions", "recent"]) == 0
    list_out = capsys.readouterr().out
    assert "#2" in list_out and "#1" in list_out

    # Choose and mutate by the exact ID from the list.
    assert main(["transaction", "edit-amount", "--id", "2", "--amount", "12"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "disposition": "UPDATED",
        "transaction_id": "2",
    }
    assert main(["transaction", "delete", "--id", "1"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "disposition": "DELETED",
        "transaction_id": "1",
    }

    # The list reflects the mutated state through the same G1 path.
    assert main(["transactions", "recent"]) == 0
    final_out = capsys.readouterr().out
    assert "🟢 #2 | +12 USDT" in final_out
    assert "#1" not in final_out
