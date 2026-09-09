"""Bounded stage-F5 tests for the standalone Hermes fast-path plugin.

Covers ``integrations/hermes/plugins/blackcat-finance-fastpath/``:

- registration: reads only the expected plugin settings through
  ``ctx.get_config``, registers exactly one ``"telegram"`` platform
  handler factory AND one ``pre_gateway_dispatch`` gateway fallback
  hook when the settings are valid, registers neither path when they
  are not, and needs neither the python-telegram-bot SDK nor the
  finance package to do so
- scoping: the PTB ``filters.MessageFilter`` installed by the platform
  factory matches only the exact configured chat, the exact configured
  topic thread, and candidate ``+<digit>`` / ``-<digit>`` text;
  unrelated messages never select the handler
- PTB runtime compatibility (F5R1): the platform factory executes
  against the ``telegram.ext`` surface used by the real Hermes host
  (``MessageHandler`` and ``filters.MessageFilter`` exported, top-level
  ``telegram.ext.MessageFilter`` NOT exported) without ImportError,
  still registers a MessageHandler, and keeps the routing semantics
- provenance: every native Telegram value reaches the finance CLI argv
  unchanged, as exactly one argv element per value
- time: ``--transaction-date`` is derived from the native Telegram
  timestamp in the configured business timezone and ``--received-at``
  stays the native aware timestamp; no wall clock is involved
- child environment: only ``HERMES_FINANCE_DB_PATH`` is added for the
  child and the parent ``os.environ`` is not mutated
- UX: the three fixed replies and the generic failure for every
  failure mode, with no internal detail exposed to Telegram
- reconnect rewiring (v1.0.1 slice B): the same registered platform
  factory safely wires successive fresh PTB Applications (initial
  connect, then a rebuilt adapter after a reconnect) and each
  application independently receives the Finance scoped handler; the
  plugin owns no mutable state that could bind the handler to the
  first adapter/application
- gateway recovery fallback (v1.0.1): the synchronous
  ``pre_gateway_dispatch`` hook is invoked in the real host keyword
  style (``event``/``gateway``/``session_store`` plus arbitrary
  additive kwargs), routes on the normalized SessionSource-like
  source (enum-like platform, canonical string chat/thread ids),
  passes all unrelated gateway traffic through untouched, and models
  the production reconnect-loss failure (native factory registered
  but never wired into the rebuilt Application) where the hook alone
  consumes the Finance candidate (action=skip before LLM dispatch),
  schedules a plugin-owned async task on the running loop, and
  drives the SAME shared ingest core with the SAME provenance rules,
  fixed replies, task lifecycle, and fail-closed scheduling
  behavior as the native path; routing and ingestion provenance are
  separate gates, so a known Finance candidate with missing or
  invalid native provenance is still consumed (skip) while the CLI
  is never invoked
- architecture: the plugin imports no Hermes internal module
  (``gateway.*``, ``hermes_cli.*``); the only Hermes dependency is the
  runtime ctx contract / hook payload
- pyproject: the mypy exclusion for the hyphenated Hermes-host plugin
  directory is anchored to exactly that directory and never
  blanket-excludes ``integrations/`` or any unrelated integration

No real Hermes, python-telegram-bot, or finance installation is
required: PTB, the gateway events, and the host context are bounded
fakes.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import logging
import os
import re
import sys
import tomllib
import zoneinfo
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pytest import MonkeyPatch

REPO_ROOT = Path(__file__).resolve().parents[1]

PLUGIN_INIT = (
    REPO_ROOT
    / "integrations"
    / "hermes"
    / "plugins"
    / "blackcat-finance-fastpath"
    / "__init__.py"
)

SETTING_KEYS = frozenset(
    {"cli_path", "database_path", "chat_id", "thread_id", "business_timezone"}
)

CHAT_ID = -1001234567890
THREAD_ID = 77

VALID_SETTINGS: dict[str, Any] = {
    "cli_path": "C:/hermes/bin/hermes-finance.exe",
    "database_path": "C:/hermes/data/finance.sqlite",
    "chat_id": CHAT_ID,
    "thread_id": THREAD_ID,
    "business_timezone": "Europe/Moscow",
}

RESPONSE_CREATED = "✅ Записано"
RESPONSE_DUPLICATE = "↩️ Уже записано"
RESPONSE_FAILURE = "⚠️ Не удалось записать операцию"

# The finance test venv deliberately ships no IANA tz database (the
# accepted finance config rejects timezone strings for exactly this
# reason), so a bounded mapping provides the two identifiers these
# tests need while every unknown key still raises the real
# ``ZoneInfoNotFoundError`` through the real ``ZoneInfo``.
_REAL_ZONEINFO = zoneinfo.ZoneInfo
_BOUNDED_IANA_ZONES: dict[str, timezone] = {
    "UTC": UTC,
    "Europe/Moscow": timezone(timedelta(hours=3)),
    "Asia/Bangkok": timezone(timedelta(hours=7)),
}


def _bounded_zoneinfo(key: str) -> tzinfo:
    """Resolve the bounded test identifiers, delegate the rest."""
    if key in _BOUNDED_IANA_ZONES:
        return _BOUNDED_IANA_ZONES[key]
    return _REAL_ZONEINFO(key)


def _install_bounded_tz_database(monkeypatch: MonkeyPatch) -> None:
    """Make ``zoneinfo.ZoneInfo`` provide the bounded test identifiers."""
    monkeypatch.setattr(zoneinfo, "ZoneInfo", _bounded_zoneinfo)


def _load_plugin(monkeypatch: MonkeyPatch) -> ModuleType:
    """Load the plugin module from its repository source location."""
    monkeypatch.delitem(sys.modules, "telegram", raising=False)
    monkeypatch.delitem(sys.modules, "telegram.ext", raising=False)
    _install_bounded_tz_database(monkeypatch)
    spec = importlib.util.spec_from_file_location(
        "blackcat_finance_fastpath", PLUGIN_INIT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The dataclass processing inside the plugin resolves its string
    # annotations through sys.modules, so the module must be registered
    # before it is executed. monkeypatch removes it again after the test.
    monkeypatch.setitem(sys.modules, "blackcat_finance_fastpath", module)
    spec.loader.exec_module(module)
    return module


@dataclass
class FakeCtx:
    """Bounded fake of the documented Hermes plugin context."""

    settings: dict[str, Any]
    read_keys: list[str] = field(default_factory=list)
    platform_registrations: list[tuple[str, Any]] = field(default_factory=list)
    hook_registrations: list[tuple[str, Any]] = field(default_factory=list)

    def get_config(self, key: str) -> Any:
        self.read_keys.append(key)
        return self.settings.get(key)

    def register_platform_handler(self, platform: str, factory: Any) -> None:
        self.platform_registrations.append((platform, factory))

    def register_hook(self, hook_name: str, hook: Any) -> None:
        self.hook_registrations.append((hook_name, hook))


class FakeMessageFilter:
    """Bounded stand-in for ``telegram.ext.filters.MessageFilter``."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name

    def filter(self, message: Any) -> bool:
        """Bounded stand-in for the PTB base-class method (no match)."""
        return False


class FakeMessageHandler:
    """Bounded stand-in for ``telegram.ext.MessageHandler``."""

    def __init__(self, message_filter: Any, callback: Any) -> None:
        self.message_filter = message_filter
        self.callback = callback


class _FakeFiltersModule:
    """Bounded fake of the ``telegram.ext.filters`` module surface.

    Real Hermes imports ``filters`` from ``telegram.ext`` and the
    supported custom-filter base is ``filters.MessageFilter``.
    """

    MessageFilter = FakeMessageFilter


class _FakeTelegramExtModule:
    """Bounded fake of the ``telegram.ext`` SDK module surface.

    Models the export surface of the PTB runtime shipped by real
    Hermes: ``MessageHandler`` and ``filters`` are available, but a
    top-level ``MessageFilter`` is NOT exported. The F5R1 runtime
    defect was an ``ImportError`` on exactly that absent name.
    """

    MessageHandler = FakeMessageHandler
    filters = _FakeFiltersModule


class _FakeTelegramModule:
    """Bounded fake of the ``telegram`` SDK package surface."""

    ext = _FakeTelegramExtModule


@dataclass
class FakeApplication:
    """Bounded fake of the native python-telegram-bot Application."""

    handlers: list[FakeMessageHandler] = field(default_factory=list)

    def add_handler(self, handler: FakeMessageHandler) -> None:
        self.handlers.append(handler)


def _install_fake_ptb(monkeypatch: MonkeyPatch) -> FakeApplication:
    """Make ``from telegram.ext import ...`` resolve to bounded fakes."""
    monkeypatch.setitem(sys.modules, "telegram", _FakeTelegramModule())
    monkeypatch.setitem(sys.modules, "telegram.ext", _FakeTelegramExtModule())
    return FakeApplication()


@dataclass
class FakeChat:
    id: int


@dataclass
class FakeTelegramMessage:
    """Bounded fake of a native Telegram text message."""

    chat: FakeChat
    message_thread_id: int | None
    message_id: int
    text: str | None
    date: datetime | None
    replies: list[str] = field(default_factory=list)

    @property
    def chat_id(self) -> int:
        return self.chat.id

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)


@dataclass
class FakeUpdate:
    """Bounded fake of a native Telegram update."""

    update_id: int
    message: FakeTelegramMessage


@dataclass
class FakePlatform:
    """Bounded stand-in for an enum-like platform identifier.

    Real Hermes ``SessionSource.platform`` is enum-like with the
    platform name in ``.value``. The field is typed ``Any`` so the
    routing tests can also model a malformed non-string value.
    """

    value: Any


@dataclass
class FakeSessionSource:
    """Bounded fake of the normalized Hermes SessionSource.

    Real contract: ``platform`` (enum-like), ``chat_id`` (canonical
    decimal string), ``thread_id`` (Optional canonical decimal
    string).
    """

    platform: Any
    chat_id: Any
    thread_id: Any


@dataclass
class FakeMessageEvent:
    """Bounded fake of the normalized Hermes gateway MessageEvent.

    Models the documented Telegram MessageEvent surface of the Hermes
    gateway: ``text``, ``source`` (the normalized SessionSource),
    ``raw_message`` (the native Telegram Message), ``message_id``
    (string form of the native message id), ``platform_update_id``
    (the real Telegram update id), and ``timestamp`` (the native
    message date).
    """

    source: Any
    text: Any
    raw_message: Any
    message_id: Any
    platform_update_id: Any
    timestamp: Any


def _telegram_source(
    *,
    chat_id: Any = None,
    thread_id: Any = None,
    platform: Any = None,
) -> FakeSessionSource:
    """Build a Telegram Finance SessionSource-like routing source."""
    return FakeSessionSource(
        platform=FakePlatform("telegram") if platform is None else platform,
        chat_id=str(CHAT_ID) if chat_id is None else chat_id,
        thread_id=str(THREAD_ID) if thread_id is None else thread_id,
    )


def _message_event(
    message: Any = None,
    *,
    source: Any = None,
    text: Any = None,
    platform_update_id: Any = 9001,
    raw_message: Any = False,
) -> FakeMessageEvent:
    """Build a normalized Telegram gateway event.

    ``source`` defaults to a Telegram Finance SessionSource-like
    double; ``raw_message=False`` means "carry the native message".
    Pass ``raw_message=None`` explicitly to model a candidate whose
    native message is absent.
    """
    native = _native_message() if message is None else message
    return FakeMessageEvent(
        source=_telegram_source() if source is None else source,
        text=native.text if text is None else text,
        raw_message=native if raw_message is False else raw_message,
        message_id=str(native.message_id),
        platform_update_id=platform_update_id,
        timestamp=native.date,
    )


def _native_message(
    *,
    text: str | None = "+25 Работа Проект A",
    date: datetime | None = None,
    chat_id: int = CHAT_ID,
    thread_id: int | None = THREAD_ID,
    message_id: int = 555,
) -> FakeTelegramMessage:
    return FakeTelegramMessage(
        chat=FakeChat(chat_id),
        message_thread_id=thread_id,
        message_id=message_id,
        text=text,
        date=date if date is not None else datetime(2026, 9, 4, 18, 30, tzinfo=UTC),
    )


@dataclass
class SpawnCall:
    """One captured finance CLI subprocess invocation."""

    argv: list[str]
    env: dict[str, str]


def _install_spawn_capture(
    monkeypatch: MonkeyPatch,
    plugin: ModuleType,
    calls: list[SpawnCall],
    result: tuple[int | None, bytes, bytes] | None = None,
    error: Exception | None = None,
) -> None:
    """Replace the plugin subprocess spawn with a bounded fake."""

    async def fake_spawn(
        argv: list[str], env: dict[str, str]
    ) -> tuple[int | None, bytes, bytes]:
        calls.append(SpawnCall(list(argv), dict(env)))
        if error is not None:
            raise error
        assert result is not None
        return result

    monkeypatch.setattr(plugin, "_spawn_process", fake_spawn)


def _registered_factory(
    monkeypatch: MonkeyPatch, settings: dict[str, Any] | None = None
) -> tuple[ModuleType, FakeCtx, Any]:
    """Load the plugin, register it, and return ctx and the factory."""
    plugin = _load_plugin(monkeypatch)
    ctx = FakeCtx(dict(VALID_SETTINGS if settings is None else settings))
    plugin.register(ctx)
    assert len(ctx.platform_registrations) == 1
    platform, factory = ctx.platform_registrations[0]
    assert platform == "telegram"
    return plugin, ctx, factory


def _make_callback(
    monkeypatch: MonkeyPatch, settings: dict[str, Any] | None = None
) -> tuple[ModuleType, Any]:
    """Register the plugin, run the factory, and return the callback."""
    plugin, _, factory = _registered_factory(monkeypatch, settings)
    application = _install_fake_ptb(monkeypatch)
    factory(application, object())
    assert len(application.handlers) == 1
    return plugin, application.handlers[0].callback


def _scoped_filter(
    monkeypatch: MonkeyPatch, settings: dict[str, Any] | None = None
) -> Any:
    """Return the PTB message filter installed by the plugin factory."""
    _, _, factory = _registered_factory(monkeypatch, settings)
    application = _install_fake_ptb(monkeypatch)
    factory(application, object())
    assert len(application.handlers) == 1
    handler = application.handlers[0]
    assert isinstance(handler.message_filter, FakeMessageFilter)
    return handler.message_filter


def _registered_hook(
    monkeypatch: MonkeyPatch, settings: dict[str, Any] | None = None
) -> tuple[ModuleType, FakeCtx, Any]:
    """Load the plugin, register it, and return ctx and the fallback hook."""
    plugin = _load_plugin(monkeypatch)
    ctx = FakeCtx(dict(VALID_SETTINGS if settings is None else settings))
    plugin.register(ctx)
    assert len(ctx.platform_registrations) == 1
    assert len(ctx.hook_registrations) == 1
    hook_name, hook = ctx.hook_registrations[0]
    assert hook_name == "pre_gateway_dispatch"
    assert callable(hook)
    return plugin, ctx, hook


def _hook_tasks(hook: Any) -> set[asyncio.Task[None]]:
    """Return the closure-owned retained-task set of the fallback hook."""
    cells = [cell.cell_contents for cell in (hook.__closure__ or [])]
    task_sets = [cell for cell in cells if isinstance(cell, set)]
    assert len(task_sets) == 1
    return task_sets[0]


async def _drain_hook_tasks(hook: Any) -> None:
    """Wait for every retained fallback task and its done callbacks.

    Task exceptions are collected (not raised): the done callback of
    the hook is the component under test for exception consumption.
    """
    for _ in range(100):
        pending = list(_hook_tasks(hook))
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.sleep(0)
    raise AssertionError("the fallback tasks did not finish")


# ------------------------------------------------------------------
# Registration (section 17)
# ------------------------------------------------------------------


def test_plugin_module_imports_no_forbidden_dependency() -> None:
    """Module-level imports stay stdlib and exclude forbidden modules."""
    tree = ast.parse(PLUGIN_INIT.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    for forbidden in ("hermes_finance", "telegram", "sys", "sqlite3", "subprocess"):
        assert forbidden not in imported


def test_plugin_source_uses_no_shell_or_system_apis() -> None:
    source = PLUGIN_INIT.read_text(encoding="utf-8")
    for forbidden in (
        "create_subprocess_shell",
        "shell=True",
        "os.system",
        "popen",
    ):
        assert forbidden not in source.lower()


def test_register_needs_no_telegram_sdk_and_imports_none(
    monkeypatch: MonkeyPatch,
) -> None:
    plugin = _load_plugin(monkeypatch)
    assert "telegram" not in sys.modules
    assert "telegram.ext" not in sys.modules
    plugin.register(FakeCtx(dict(VALID_SETTINGS)))
    assert "telegram" not in sys.modules
    assert "telegram.ext" not in sys.modules


def test_register_reads_only_the_expected_plugin_settings(
    monkeypatch: MonkeyPatch,
) -> None:
    _, ctx, _ = _registered_factory(monkeypatch)
    assert set(ctx.read_keys) == SETTING_KEYS


def test_factory_installs_exactly_one_scoped_message_handler(
    monkeypatch: MonkeyPatch,
) -> None:
    _, _, factory = _registered_factory(monkeypatch)
    application = _install_fake_ptb(monkeypatch)
    factory(application, object())
    assert len(application.handlers) == 1


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("cli_path", None),
        ("cli_path", ""),
        ("cli_path", "   "),
        ("cli_path", 123),
        ("database_path", None),
        ("database_path", ""),
        ("database_path", "  "),
        ("chat_id", None),
        ("chat_id", True),
        ("chat_id", 0),
        ("chat_id", "-1001234567890"),
        ("chat_id", -1001234567890.0),
        ("thread_id", None),
        ("thread_id", True),
        ("thread_id", 0),
        ("thread_id", -77),
        ("thread_id", 77.0),
        ("business_timezone", None),
        ("business_timezone", ""),
        ("business_timezone", "   "),
        ("business_timezone", "Not/AZone"),
    ],
)
def test_register_fails_closed_on_invalid_setting(
    monkeypatch: MonkeyPatch, key: str, value: Any
) -> None:
    plugin = _load_plugin(monkeypatch)
    settings = dict(VALID_SETTINGS)
    settings[key] = value
    ctx = FakeCtx(settings)
    plugin.register(ctx)
    assert ctx.platform_registrations == []
    assert ctx.hook_registrations == []
    assert "telegram" not in sys.modules


def test_missing_setting_fails_closed(monkeypatch: MonkeyPatch) -> None:
    plugin = _load_plugin(monkeypatch)
    settings = dict(VALID_SETTINGS)
    del settings["database_path"]
    ctx = FakeCtx(settings)
    plugin.register(ctx)
    assert ctx.platform_registrations == []
    assert ctx.hook_registrations == []
    assert "telegram" not in sys.modules


# ------------------------------------------------------------------
# Scoped message handler (section 18)
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["+25 Работа Проект A", "-10 Инфраструктура Хостинг", "+0.25 Сервисы Подписка"],
)
def test_filter_matches_finance_candidates_in_the_finance_topic(
    monkeypatch: MonkeyPatch, text: str
) -> None:
    message_filter = _scoped_filter(monkeypatch)
    assert message_filter.filter(_native_message(text=text)) is True


@pytest.mark.parametrize(
    "text",
    [
        "Покажи отчёт за август",
        "Сколько заработал?",
        "Привет",
        "+ покажи доход",
        "25 Работа Проект A",
        "окей +25",
        "",
    ],
)
def test_filter_does_not_match_normal_text_in_the_finance_topic(
    monkeypatch: MonkeyPatch, text: str
) -> None:
    message_filter = _scoped_filter(monkeypatch)
    assert message_filter.filter(_native_message(text=text)) is False


def test_filter_does_not_match_another_chat(monkeypatch: MonkeyPatch) -> None:
    message_filter = _scoped_filter(monkeypatch)
    message = _native_message(text="+25 Работа Проект A", chat_id=-1009999999999)
    assert message_filter.filter(message) is False


def test_filter_does_not_match_another_topic(monkeypatch: MonkeyPatch) -> None:
    message_filter = _scoped_filter(monkeypatch)
    message = _native_message(text="+25 Работа Проект A", thread_id=99)
    assert message_filter.filter(message) is False


def test_filter_does_not_match_absent_thread(monkeypatch: MonkeyPatch) -> None:
    message_filter = _scoped_filter(monkeypatch)
    message = _native_message(text="+25 Работа Проект A", thread_id=None)
    assert message_filter.filter(message) is False


def test_filter_does_not_match_non_text(monkeypatch: MonkeyPatch) -> None:
    message_filter = _scoped_filter(monkeypatch)
    message = _native_message(text=None)
    assert message_filter.filter(message) is False


def test_filter_does_not_match_message_without_chat_id(
    monkeypatch: MonkeyPatch,
) -> None:
    message_filter = _scoped_filter(monkeypatch)
    message = _native_message(text="+25 Работа Проект A")
    object.__setattr__(message, "chat", None)
    assert message_filter.filter(message) is False


# ------------------------------------------------------------------
# PTB runtime compatibility (F5R1 regression)
# ------------------------------------------------------------------


def test_factory_runs_on_the_real_hermes_ptb_export_surface(
    monkeypatch: MonkeyPatch,
) -> None:
    """F5R1: the factory needs no top-level ``telegram.ext.MessageFilter``.

    The real Hermes Agent v0.21.0 PTB runtime exports ``MessageHandler``
    and ``filters.MessageFilter`` but NOT a top-level
    ``telegram.ext.MessageFilter``; the previously accepted plugin
    raised ``ImportError`` from inside the platform factory on exactly
    that surface. This test executes the real registered factory
    against that exact export surface and verifies the full behavioral
    outcome: no ImportError, a ``filters.MessageFilter``-derived
    Finance filter, one registered ``MessageHandler``, and unchanged
    matching semantics.
    """
    _, _, factory = _registered_factory(monkeypatch)
    application = _install_fake_ptb(monkeypatch)
    ext_module = sys.modules["telegram.ext"]

    # Sanity: the fake really models the real Hermes PTB surface the
    # defect was observed on — the top-level name is absent while the
    # supported base and the handler class are present.
    assert not hasattr(ext_module, "MessageFilter")
    assert ext_module.MessageHandler is FakeMessageHandler
    assert ext_module.filters.MessageFilter is FakeMessageFilter

    # The real platform factory must execute without ImportError.
    factory(application, object())

    # Exactly one MessageHandler is registered with an instantiable
    # filters.MessageFilter-derived custom Finance filter.
    assert len(application.handlers) == 1
    handler = application.handlers[0]
    assert isinstance(handler, FakeMessageHandler)
    assert isinstance(handler.message_filter, FakeMessageFilter)

    # Exact chat/thread/+digit/-digit matching still works.
    message_filter = handler.message_filter
    assert message_filter.filter(_native_message(text="+25 Работа Проект A")) is True
    assert message_filter.filter(_native_message(text="-10 Прокат")) is True

    # Unrelated messages still do not match.
    assert message_filter.filter(_native_message(text="Сколько заработал?")) is False
    assert (
        message_filter.filter(
            _native_message(text="+25 Работа Проект A", chat_id=-1009999999999)
        )
        is False
    )
    assert (
        message_filter.filter(_native_message(text="+25 Работа Проект A", thread_id=99))
        is False
    )
    assert (
        message_filter.filter(
            _native_message(text="+25 Работа Проект A", thread_id=None)
        )
        is False
    )


# ------------------------------------------------------------------
# Callback provenance (section 19)
# ------------------------------------------------------------------


def test_callback_passes_every_native_field_to_the_cli_unchanged(
    monkeypatch: MonkeyPatch,
) -> None:
    plugin, callback = _make_callback(monkeypatch)
    calls: list[SpawnCall] = []
    _install_spawn_capture(
        monkeypatch,
        plugin,
        calls,
        result=(0, b'{"disposition": "CREATED", "transaction_id": "t-049"}', b""),
    )
    message = _native_message(text="+25 Работа Проект A")
    update = FakeUpdate(update_id=9001, message=message)
    asyncio.run(callback(update, None))
    assert len(calls) == 1
    argv = calls[0].argv
    assert argv == [
        VALID_SETTINGS["cli_path"],
        "ingest-telegram",
        "--text",
        "+25 Работа Проект A",
        "--chat-id",
        str(CHAT_ID),
        "--thread-id",
        str(THREAD_ID),
        "--message-id",
        "555",
        "--update-id",
        "9001",
        "--transaction-date",
        "2026-09-04",
        "--received-at",
        "2026-09-04T18:30:00+00:00",
    ]
    # The exact message text is one single unchanged argv element.
    assert argv[argv.index("--text") + 1] == "+25 Работа Проект A"
    assert message.replies == [RESPONSE_CREATED]


def _broken_provenance_update(field_name: str) -> FakeUpdate:
    """Build a native update whose ``field_name`` provenance is absent."""
    message = _native_message()
    update = FakeUpdate(9001, message)
    target: Any = update if field_name == "update_id" else message
    object.__setattr__(target, field_name, None)
    return update


@pytest.mark.parametrize(
    "field_name", ["date", "message_thread_id", "text", "message_id", "update_id"]
)
def test_callback_skips_the_cli_when_native_provenance_is_absent(
    monkeypatch: MonkeyPatch, field_name: str
) -> None:
    plugin, callback = _make_callback(monkeypatch)
    calls: list[SpawnCall] = []
    _install_spawn_capture(monkeypatch, plugin, calls)
    update = _broken_provenance_update(field_name)
    asyncio.run(callback(update, None))
    assert calls == []
    assert update.message.replies == [RESPONSE_FAILURE]


def test_callback_skips_the_cli_for_a_naive_message_date(
    monkeypatch: MonkeyPatch,
) -> None:
    plugin, callback = _make_callback(monkeypatch)
    calls: list[SpawnCall] = []
    _install_spawn_capture(monkeypatch, plugin, calls)
    # A naive Telegram timestamp must be rejected: strip the offset from
    # an aware one instead of constructing a naive datetime directly.
    message = _native_message(
        date=datetime(2026, 9, 4, 18, 30, tzinfo=UTC).replace(tzinfo=None)
    )
    asyncio.run(callback(FakeUpdate(9001, message), None))
    assert calls == []
    assert message.replies == [RESPONSE_FAILURE]


# ------------------------------------------------------------------
# Business timezone (section 20)
# ------------------------------------------------------------------


def test_transaction_date_uses_business_timezone_not_utc(
    monkeypatch: MonkeyPatch,
) -> None:
    # 2026-09-04 23:30 UTC is already 2026-09-05 02:30 in Moscow.
    plugin, callback = _make_callback(
        monkeypatch, dict(VALID_SETTINGS, business_timezone="Europe/Moscow")
    )
    calls: list[SpawnCall] = []
    _install_spawn_capture(
        monkeypatch,
        plugin,
        calls,
        result=(0, b'{"disposition": "CREATED", "transaction_id": "t-1"}', b""),
    )
    message = _native_message(date=datetime(2026, 9, 4, 23, 30, tzinfo=UTC))
    asyncio.run(callback(FakeUpdate(9001, message), None))
    argv = calls[0].argv
    assert argv[argv.index("--transaction-date") + 1] == "2026-09-05"
    assert argv[argv.index("--received-at") + 1] == "2026-09-04T23:30:00+00:00"


# ------------------------------------------------------------------
# Child environment (section 21)
# ------------------------------------------------------------------


def test_child_env_gets_only_the_configured_database_path(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("HERMES_FINANCE_DB_PATH", raising=False)
    plugin, callback = _make_callback(monkeypatch)
    calls: list[SpawnCall] = []
    _install_spawn_capture(
        monkeypatch,
        plugin,
        calls,
        result=(0, b'{"disposition": "CREATED", "transaction_id": "t-1"}', b""),
    )
    parent_before = dict(os.environ)
    asyncio.run(callback(FakeUpdate(9001, _native_message()), None))
    child_env = calls[0].env
    assert child_env["HERMES_FINANCE_DB_PATH"] == VALID_SETTINGS["database_path"]
    assert child_env == {
        **os.environ,
        "HERMES_FINANCE_DB_PATH": VALID_SETTINGS["database_path"],
    }
    assert dict(os.environ) == parent_before
    assert "HERMES_FINANCE_DB_PATH" not in os.environ


# ------------------------------------------------------------------
# Result mapping and UX (section 22)
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("disposition", "expected_reply"),
    [
        ("CREATED", RESPONSE_CREATED),
        ("DUPLICATE_MESSAGE", RESPONSE_DUPLICATE),
        ("DUPLICATE_UPDATE", RESPONSE_DUPLICATE),
    ],
)
def test_reply_for_recognized_dispositions(
    monkeypatch: MonkeyPatch, disposition: str, expected_reply: str
) -> None:
    plugin, callback = _make_callback(monkeypatch)
    stdout = json.dumps(
        {"disposition": disposition, "transaction_id": "t-1"}
    ).encode("utf-8")
    calls: list[SpawnCall] = []
    _install_spawn_capture(monkeypatch, plugin, calls, result=(0, stdout, b""))
    message = _native_message()
    asyncio.run(callback(FakeUpdate(9001, message), None))
    assert len(calls) == 1
    assert message.replies == [expected_reply]


@pytest.mark.parametrize(
    "result",
    [
        (1, b"", b"Traceback (most recent call last):\n  File \"C:/secret/db.sqlite\""),
        (2, b"not json at all", b""),
        (0, b'{"disposition": "REJECTED"}', b""),
        (0, b'{"no_disposition": true}', b""),
    ],
)
def test_reply_is_the_generic_failure_for_every_failure_mode(
    monkeypatch: MonkeyPatch, result: tuple[int | None, bytes, bytes]
) -> None:
    plugin, callback = _make_callback(monkeypatch)
    calls: list[SpawnCall] = []
    _install_spawn_capture(monkeypatch, plugin, calls, result=result)
    message = _native_message()
    asyncio.run(callback(FakeUpdate(9001, message), None))
    assert len(calls) == 1
    assert message.replies == [RESPONSE_FAILURE]


def test_reply_is_the_generic_failure_when_the_cli_cannot_start(
    monkeypatch: MonkeyPatch,
) -> None:
    plugin, callback = _make_callback(monkeypatch)
    calls: list[SpawnCall] = []
    _install_spawn_capture(monkeypatch, plugin, calls, error=OSError("spawn boom"))
    message = _native_message()
    asyncio.run(callback(FakeUpdate(9001, message), None))
    # No automatic retry: the finance CLI is attempted exactly once.
    assert len(calls) == 1
    assert message.replies == [RESPONSE_FAILURE]


def test_failure_reply_never_exposes_internals(monkeypatch: MonkeyPatch) -> None:
    plugin, callback = _make_callback(monkeypatch)
    secret_stderr = (
        b"Traceback: boom at C:/hermes/data/finance.sqlite while running"
        b" hermes-finance ingest-telegram"
    )
    calls: list[SpawnCall] = []
    _install_spawn_capture(monkeypatch, plugin, calls, result=(1, b"", secret_stderr))
    message = _native_message()
    asyncio.run(callback(FakeUpdate(9001, message), None))
    reply = message.replies[0]
    assert reply == RESPONSE_FAILURE
    for internal in ("Traceback", "finance.sqlite", "hermes-finance", "ingest-telegram"):
        assert internal not in reply


# ------------------------------------------------------------------
# Narrow mypy exclusion (pyproject remediation)
# ------------------------------------------------------------------


def test_mypy_exclusion_is_scoped_to_the_fastpath_plugin_only() -> None:
    """The mypy exclude matches only the plugin directory, nothing else.

    Reads the actual ``pyproject.toml`` mypy exclusion, compiles it,
    and verifies with both POSIX and Windows separators that it matches
    exactly the ``blackcat-finance-fastpath`` plugin source directory
    while ``integrations/`` as a whole, ``integrations/hermes/``,
    ``integrations/hermes/plugins/``, sibling integrations, the typed
    finance package, and the tests all stay eligible for mypy.
    """
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    mypy_config = config["tool"]["mypy"]
    exclude = mypy_config["exclude"]
    assert isinstance(exclude, list)
    assert len(exclude) == 1
    assert isinstance(exclude[0], str)
    pattern = re.compile(exclude[0])

    plugin_files = [
        "integrations/hermes/plugins/blackcat-finance-fastpath/__init__.py",
        "integrations/hermes/plugins/blackcat-finance-fastpath/plugin.yaml",
    ]
    unrelated_files = [
        "integrations/hermes/plugins.py",
        "integrations/hermes/plugins/other-plugin/__init__.py",
        "integrations/hermes/plugins/blackcat-finance-fastpath-extra/__init__.py",
        "integrations/hermes/skills/blackcat-finance/helper.py",
        "integrations/hermes/utils.py",
        "integrations/other/tool.py",
        "src/hermes_finance/cli.py",
        "tests/test_cli.py",
    ]
    for path in plugin_files:
        assert pattern.search(path) is not None, path
        assert pattern.search(path.replace("/", "\\")) is not None, path
    for path in unrelated_files:
        assert pattern.search(path) is None, path
        assert pattern.search(path.replace("/", "\\")) is None, path


def test_mypy_exclusion_keeps_finance_and_tests_eligible() -> None:
    """No exclusion pattern may match the typed finance surface."""
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    exclude = config["tool"]["mypy"]["exclude"]
    assert isinstance(exclude, list)
    patterns = [re.compile(expr) for expr in exclude if isinstance(expr, str)]
    assert len(patterns) == len(exclude)
    for typed_path in ("src/hermes_finance/cli.py", "tests/test_hermes_skill.py"):
        for pattern in patterns:
            assert pattern.search(typed_path) is None, (pattern.pattern, typed_path)


# ------------------------------------------------------------------
# QA RED remediation regression tests
# ------------------------------------------------------------------


def _run_callback(
    monkeypatch: MonkeyPatch,
    update: FakeUpdate,
    result: tuple[int | None, bytes, bytes] | None = None,
    error: Exception | None = None,
    settings: dict[str, Any] | None = None,
) -> list[SpawnCall]:
    """Run one callback invocation and return the captured CLI calls."""
    plugin, callback = _make_callback(monkeypatch, settings)
    calls: list[SpawnCall] = []
    _install_spawn_capture(monkeypatch, plugin, calls, result=result, error=error)
    asyncio.run(callback(update, None))
    return calls


def _invalid_provenance_update(field: str, value: Any) -> FakeUpdate:
    """Build a native update whose ``field`` provenance is ``value``."""
    message = FakeTelegramMessage(
        chat=FakeChat(value if field == "chat_id" else CHAT_ID),
        message_thread_id=value if field == "thread_id" else THREAD_ID,
        message_id=value if field == "message_id" else 555,
        text="+25 Работа Проект A",
        date=datetime(2026, 9, 4, 18, 30, tzinfo=UTC),
    )
    return FakeUpdate(
        value if field == "update_id" else 9001,
        message,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("message_id", 0),
        ("message_id", -1),
        ("update_id", -1),
        ("chat_id", True),
        ("thread_id", True),
        ("message_id", True),
        ("update_id", True),
        ("chat_id", 0),
        ("thread_id", 0),
        ("thread_id", -77),
    ],
)
def test_invalid_native_provenance_never_invokes_the_cli(
    monkeypatch: MonkeyPatch, field: str, value: Any
) -> None:
    """Out-of-range or bool ids fail closed before any subprocess."""
    update = _invalid_provenance_update(field, value)
    message = update.message
    calls = _run_callback(
        monkeypatch,
        update,
        result=(0, b'{"disposition": "CREATED", "transaction_id": "t-1"}', b""),
    )
    assert calls == []
    assert message.replies == [RESPONSE_FAILURE]


@pytest.mark.parametrize("disposition", ["CREATED", "DUPLICATE_MESSAGE"])
@pytest.mark.parametrize(
    "transaction_id",
    [None, "", "   ", 42, 3.14, ["t-1"], {"id": "t-1"}, True],
)
def test_invalid_transaction_id_yields_generic_failure(
    monkeypatch: MonkeyPatch, disposition: str, transaction_id: Any
) -> None:
    """A success reply requires a non-empty string transaction_id."""
    stdout = json.dumps(
        {"disposition": disposition, "transaction_id": transaction_id}
    ).encode("utf-8")
    message = _native_message()
    calls = _run_callback(
        monkeypatch, FakeUpdate(9001, message), result=(0, stdout, b"")
    )
    assert len(calls) == 1
    assert message.replies == [RESPONSE_FAILURE]


@pytest.mark.parametrize("payload", [{"disposition": []}, {"disposition": {}}])
def test_malformed_disposition_type_never_raises(
    monkeypatch: MonkeyPatch, payload: dict[str, Any]
) -> None:
    """A list or object disposition is a generic failure, never a TypeError."""
    stdout = json.dumps(payload).encode("utf-8")
    message = _native_message()
    calls = _run_callback(
        monkeypatch, FakeUpdate(9001, message), result=(0, stdout, b"")
    )
    assert len(calls) == 1
    assert message.replies == [RESPONSE_FAILURE]


@pytest.mark.parametrize("disposition", [None, 7, ["CREATED"], {"x": 1}, "REJECTED"])
def test_non_string_or_unrecognized_disposition_yields_generic_failure(
    monkeypatch: MonkeyPatch, disposition: Any
) -> None:
    stdout = json.dumps({"disposition": disposition, "transaction_id": "t-1"}).encode(
        "utf-8"
    )
    message = _native_message()
    calls = _run_callback(
        monkeypatch, FakeUpdate(9001, message), result=(0, stdout, b"")
    )
    assert len(calls) == 1
    assert message.replies == [RESPONSE_FAILURE]


def test_nonzero_exit_log_contains_no_raw_stderr(
    monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Raw child stderr never reaches the local log."""
    stderr_sentinel = b"SECRET-STDERR-SENTINEL C:/secret/finance.sqlite"
    with caplog.at_level(logging.ERROR):
        message = _native_message()
        calls = _run_callback(
            monkeypatch,
            FakeUpdate(9001, message),
            result=(1, b"", stderr_sentinel),
        )
    assert len(calls) == 1
    assert message.replies == [RESPONSE_FAILURE]
    log_text = caplog.text
    assert "SECRET-STDERR-SENTINEL" not in log_text
    assert "finance.sqlite" not in log_text


def test_malformed_output_log_contains_no_raw_stdout(
    monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Raw child stdout never reaches the local log."""
    stdout_sentinel = b"SECRET-STDOUT-SENTINEL C:/secret/finance.sqlite"
    with caplog.at_level(logging.ERROR):
        message = _native_message()
        calls = _run_callback(
            monkeypatch,
            FakeUpdate(9001, message),
            result=(0, stdout_sentinel, b""),
        )
    assert len(calls) == 1
    assert message.replies == [RESPONSE_FAILURE]
    assert "SECRET-STDOUT-SENTINEL" not in caplog.text


def test_start_failure_log_contains_only_safe_details(
    monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A start failure logs the exception TYPE only, no message/traceback."""
    error = OSError("SECRET-START-SENTINEL C:/secret/cli-path boom")
    with caplog.at_level(logging.ERROR):
        message = _native_message()
        calls = _run_callback(monkeypatch, FakeUpdate(9001, message), error=error)
    assert len(calls) == 1
    assert message.replies == [RESPONSE_FAILURE]
    log_text = caplog.text
    assert "SECRET-START-SENTINEL" not in log_text
    assert "cli-path" not in log_text
    # The exception TYPE is the one safe detail that must be present.
    assert "OSError" in log_text
    assert "Traceback" not in log_text


def test_invalid_provenance_log_contains_no_message_content(
    monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The provenance diagnostic is value-free, including message text."""
    with caplog.at_level(logging.ERROR):
        message = _native_message(text="+25 SECRET-TEXT-SENTINEL Проект A")
        update = FakeUpdate(9001, message)
        object.__setattr__(message, "message_id", 0)
        calls = _run_callback(
            monkeypatch,
            update,
            result=(0, b'{"disposition": "CREATED", "transaction_id": "t-1"}', b""),
        )
    assert calls == []
    assert message.replies == [RESPONSE_FAILURE]
    assert "SECRET-TEXT-SENTINEL" not in caplog.text


def test_manifest_uses_manifest_version_2_without_apiversion() -> None:
    """The native manifest uses ``manifest_version: 2`` only."""
    manifest = (PLUGIN_INIT.parent / "plugin.yaml").read_text(encoding="utf-8")
    assert "apiVersion" not in manifest
    match = re.search(r"(?m)^manifest_version:\s*(\d+)\s*$", manifest)
    assert match is not None
    assert match.group(1) == "2"


# ------------------------------------------------------------------
# Reconnect rewiring (v1.0.1 slice B)
# ------------------------------------------------------------------


def _two_wired_applications(
    monkeypatch: MonkeyPatch,
) -> tuple[ModuleType, FakeApplication, FakeApplication]:
    """Register the plugin once, then wire the factory into two fresh
    Applications, modeling the Hermes reconnect lifecycle.

    The initial ``TelegramAdapter.connect()`` invokes the registered
    factory against Application A. After a fatal platform error the
    reconnect runner rebuilds the adapter, whose ``connect()`` invokes
    the SAME registered factory against a fresh Application B.
    """
    plugin, _, factory = _registered_factory(monkeypatch)
    application_a = _install_fake_ptb(monkeypatch)
    factory(application_a, object())
    application_b = _install_fake_ptb(monkeypatch)
    factory(application_b, object())
    return plugin, application_a, application_b


def test_factory_rewires_successive_fresh_applications(
    monkeypatch: MonkeyPatch,
) -> None:
    """The same registered factory wires both the initial and the
    rebuilt (reconnect) Application, each with exactly one handler."""
    _, application_a, application_b = _two_wired_applications(monkeypatch)
    assert len(application_a.handlers) == 1
    assert len(application_b.handlers) == 1


def test_rewired_handlers_are_independent_objects(
    monkeypatch: MonkeyPatch,
) -> None:
    """No handler object, filter, or callback is shared or re-bound
    between the initial Application and the reconnect Application."""
    _, application_a, application_b = _two_wired_applications(monkeypatch)
    handler_a = application_a.handlers[0]
    handler_b = application_b.handlers[0]
    assert handler_a is not handler_b
    assert handler_a.message_filter is not handler_b.message_filter
    assert handler_a.callback is not handler_b.callback
    # Rewiring the reconnect Application does not mutate the initial one.
    assert len(application_a.handlers) == 1
    assert application_a.handlers[0] is handler_a


def test_both_wired_handlers_keep_the_scoped_routing_semantics(
    monkeypatch: MonkeyPatch,
) -> None:
    """The handler on each Application independently matches only the
    exact configured Finance chat/topic with a candidate prefix."""
    _, application_a, application_b = _two_wired_applications(monkeypatch)
    for application in (application_a, application_b):
        message_filter = application.handlers[0].message_filter
        assert message_filter.filter(_native_message()) is True
        assert (
            message_filter.filter(
                _native_message(text="+25 Работа Проект A", chat_id=-1009999999999)
            )
            is False
        )
        assert (
            message_filter.filter(_native_message(text="+25 Работа Проект A", thread_id=99))
            is False
        )
        assert message_filter.filter(_native_message(text="Сколько заработал?")) is False


def test_rewired_callback_invokes_the_cli_from_the_fresh_application(
    monkeypatch: MonkeyPatch,
) -> None:
    """A message routed through the reconnect Application's handler
    reaches the finance CLI exactly like on the initial Application."""
    plugin, application_a, application_b = _two_wired_applications(monkeypatch)
    calls: list[SpawnCall] = []
    _install_spawn_capture(
        monkeypatch,
        plugin,
        calls,
        result=(0, b'{"disposition": "CREATED", "transaction_id": "t-2"}', b""),
    )
    message = _native_message()
    asyncio.run(application_b.handlers[0].callback(FakeUpdate(9002, message), None))
    assert len(calls) == 1
    assert message.replies == [RESPONSE_CREATED]
    # The initial application's callback is equally alive and independent.
    message_a = _native_message(message_id=556)
    asyncio.run(application_a.handlers[0].callback(FakeUpdate(9003, message_a), None))
    assert len(calls) == 2
    assert message_a.replies == [RESPONSE_CREATED]


def test_plugin_owns_no_rebindable_module_state() -> None:
    """No function in the plugin rebinds or declares module-level state.

    The platform factory must stay a pure function of its captured
    frozen config plus the passed application: there is no ``global``/
    ``nonlocal`` rebinding anywhere and no module-level assignment
    through an attribute or subscript target that a factory invocation
    could accumulate per-adapter state in. This is the structural half
    of the proof that the plugin cannot bind its handler to the first
    adapter/application.
    """
    tree = ast.parse(PLUGIN_INIT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        assert not isinstance(node, ast.Global)
        assert not isinstance(node, ast.Nonlocal)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                assert isinstance(target, ast.Name)
        elif isinstance(node, ast.AnnAssign):
            assert isinstance(node.target, ast.Name)


def test_factory_configuration_is_captured_in_a_frozen_dataclass(
    monkeypatch: MonkeyPatch,
) -> None:
    """The only state the factory closes over is immutable.

    ``_make_platform_factory`` captures exactly the validated frozen
    ``_FastPathConfig``; the config cannot be mutated after capture, so
    successive factory invocations against fresh Applications are
    behaviorally identical.
    """
    plugin = _load_plugin(monkeypatch)
    ctx = FakeCtx(dict(VALID_SETTINGS))
    plugin.register(ctx)
    assert len(ctx.platform_registrations) == 1
    factory = ctx.platform_registrations[0][1]
    closure_cells = [c.cell_contents for c in (factory.__closure__ or [])]
    configs = [value for value in closure_cells if type(value) is plugin._FastPathConfig]
    assert len(closure_cells) == 1
    assert len(configs) == 1
    assert configs[0].__dataclass_params__.frozen


# ------------------------------------------------------------------
# Gateway recovery fallback: registration (v1.0.1)
# ------------------------------------------------------------------


def test_register_keeps_exactly_one_telegram_platform_factory(
    monkeypatch: MonkeyPatch,
) -> None:
    """The PRIMARY path stays exactly one telegram platform factory."""
    plugin = _load_plugin(monkeypatch)
    ctx = FakeCtx(dict(VALID_SETTINGS))
    plugin.register(ctx)
    assert len(ctx.platform_registrations) == 1
    platform, factory = ctx.platform_registrations[0]
    assert platform == "telegram"
    assert callable(factory)


def test_register_also_registers_one_pre_gateway_dispatch_hook(
    monkeypatch: MonkeyPatch,
) -> None:
    """The FAIL-SAFE path is exactly one pre_gateway_dispatch hook."""
    plugin = _load_plugin(monkeypatch)
    ctx = FakeCtx(dict(VALID_SETTINGS))
    plugin.register(ctx)
    assert len(ctx.hook_registrations) == 1
    hook_name, hook = ctx.hook_registrations[0]
    assert hook_name == "pre_gateway_dispatch"
    assert callable(hook)


def test_invalid_config_registers_neither_path(monkeypatch: MonkeyPatch) -> None:
    """One invalid setting disables BOTH the native handler and the hook."""
    plugin = _load_plugin(monkeypatch)
    ctx = FakeCtx(dict(VALID_SETTINGS, chat_id=0))
    plugin.register(ctx)
    assert ctx.platform_registrations == []
    assert ctx.hook_registrations == []


# ------------------------------------------------------------------
# Gateway recovery fallback: host hook contract
# ------------------------------------------------------------------


def test_fallback_hook_tolerates_real_host_style_additive_kwargs(
    monkeypatch: MonkeyPatch,
) -> None:
    """The registered hook accepts the real host keyword payload.

    Real Hermes invokes ``pre_gateway_dispatch`` with keyword
    arguments (``event``, ``gateway``, ``session_store``, ...) whose
    public contract is additive: the hook must never raise TypeError
    on current or future extra host kwargs.
    """
    plugin, _, hook = _registered_hook(monkeypatch)
    calls: list[SpawnCall] = []
    _install_spawn_capture(
        monkeypatch,
        plugin,
        calls,
        result=(0, b'{"disposition": "CREATED", "transaction_id": "t-1"}', b""),
    )
    message = _native_message()

    async def scenario() -> None:
        result = hook(
            event=_message_event(message=message),
            gateway=object(),
            session_store=object(),
            future_additive_field=object(),
        )
        assert result == {"action": "skip", "reason": "blackcat-finance"}
        await _drain_hook_tasks(hook)

    asyncio.run(scenario())
    assert len(calls) == 1
    assert message.replies == [RESPONSE_CREATED]
    assert _hook_tasks(hook) == set()


def test_fallback_hook_fails_safely_on_missing_or_malformed_event(
    monkeypatch: MonkeyPatch,
) -> None:
    """A missing or attribute-less event passes through without raising."""
    plugin, _, hook = _registered_hook(monkeypatch)
    calls: list[SpawnCall] = []
    _install_spawn_capture(monkeypatch, plugin, calls)
    assert hook() is None
    assert hook(event=None, gateway=object()) is None
    assert hook(event=object(), gateway=object(), session_store=object()) is None
    assert calls == []
    assert _hook_tasks(hook) == set()


# ------------------------------------------------------------------
# Gateway recovery fallback: routing (unrelated traffic passes through)
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("platform", "is_telegram"),
    [
        (FakePlatform("telegram"), True),
        ("telegram", True),
        (FakePlatform("discord"), False),
        ("discord", False),
        ("TELEGRAM", False),
        (None, False),
        (FakePlatform(None), False),
        (12345, False),
    ],
)
def test_fallback_hook_routes_only_telegram_platforms(
    monkeypatch: MonkeyPatch, platform: Any, is_telegram: bool
) -> None:
    """Only Telegram routes: enum-like ``.value`` or the raw string."""
    plugin, _, hook = _registered_hook(monkeypatch)
    calls: list[SpawnCall] = []
    _install_spawn_capture(monkeypatch, plugin, calls)
    # Built directly: ``None`` must stay the literal platform value,
    # not the ``_telegram_source`` default sentinel.
    source = FakeSessionSource(platform=platform, chat_id=str(CHAT_ID), thread_id=str(THREAD_ID))

    async def scenario() -> None:
        result = hook(event=_message_event(source=source), gateway=object())
        if is_telegram:
            assert result == {"action": "skip", "reason": "blackcat-finance"}
            await _drain_hook_tasks(hook)
        else:
            assert result is None

    asyncio.run(scenario())
    assert len(calls) == (1 if is_telegram else 0)
    assert _hook_tasks(hook) == set()


@pytest.mark.parametrize(
    ("chat_id", "thread_id", "is_finance"),
    [
        (CHAT_ID, THREAD_ID, True),
        (str(CHAT_ID), str(THREAD_ID), True),
        (CHAT_ID, str(THREAD_ID), True),
        (str(CHAT_ID), THREAD_ID, True),
        (-1009999999999, THREAD_ID, False),
        (str(-1009999999999), THREAD_ID, False),
        (" -1001234567890", THREAD_ID, False),
        ("-1001234567890 ", THREAD_ID, False),
        ("-1001234567890.0", THREAD_ID, False),
        (True, THREAD_ID, False),
        (CHAT_ID, True, False),
        (CHAT_ID, None, False),
        (CHAT_ID, 77.0, False),
        (CHAT_ID, "077", False),
        (CHAT_ID, " 77", False),
        (CHAT_ID, "77x", False),
    ],
)
def test_fallback_hook_routes_only_exact_finance_ids(
    monkeypatch: MonkeyPatch, chat_id: Any, thread_id: Any, is_finance: bool
) -> None:
    """A normalized id matches only as the exact int or its canonical
    decimal string; bools, floats, padded, and malformed values never
    route (negative chat ids and positive thread ids both covered)."""
    plugin, _, hook = _registered_hook(monkeypatch)
    calls: list[SpawnCall] = []
    _install_spawn_capture(monkeypatch, plugin, calls)
    # Built directly: ``None`` must stay the literal thread value,
    # not the ``_telegram_source`` default sentinel.
    source = FakeSessionSource(
        platform=FakePlatform("telegram"), chat_id=chat_id, thread_id=thread_id
    )

    async def scenario() -> None:
        result = hook(event=_message_event(source=source), gateway=object())
        if is_finance:
            assert result == {"action": "skip", "reason": "blackcat-finance"}
            await _drain_hook_tasks(hook)
        else:
            assert result is None

    asyncio.run(scenario())
    assert len(calls) == (1 if is_finance else 0)
    assert _hook_tasks(hook) == set()


@pytest.mark.parametrize(
    "text",
    [
        "Покажи отчёт за август",
        "Сколько заработал?",
        "Привет",
        "+ покажи доход",
        "25 Работа Проект A",
        "окей +25",
        "",
    ],
)
def test_fallback_hook_passes_through_ordinary_text(
    monkeypatch: MonkeyPatch, text: str
) -> None:
    plugin, _, hook = _registered_hook(monkeypatch)
    calls: list[SpawnCall] = []
    _install_spawn_capture(monkeypatch, plugin, calls)
    assert hook(event=_message_event(text=text), gateway=object()) is None
    assert calls == []
    assert _hook_tasks(hook) == set()


# ------------------------------------------------------------------
# Gateway recovery fallback: reconnect-loss simulation (v1.0.1)
# ------------------------------------------------------------------


def test_reconnect_loss_fallback_consumes_and_ingests_the_candidate(
    monkeypatch: MonkeyPatch,
) -> None:
    """Model the production failure end to end.

    Plugin registration succeeds, but after the Telegram adapter
    reconnect the rebuilt Application never wires the registered
    native platform factory, so ONLY the pre_gateway_dispatch hook
    stands between the Finance candidate and the LLM. The hook is
    invoked in the real host keyword style with a SessionSource-like
    normalized source, must consume the candidate (action=skip), and
    must process it exactly like the native handler would have.
    """
    plugin, ctx, hook = _registered_hook(monkeypatch)
    # The native platform factory exists but is deliberately NOT
    # wired into any rebuilt Application (the reconnect defect being
    # contained).
    assert len(ctx.platform_registrations) == 1
    calls: list[SpawnCall] = []
    _install_spawn_capture(
        monkeypatch,
        plugin,
        calls,
        result=(0, b'{"disposition": "CREATED", "transaction_id": "t-9"}', b""),
    )
    message = _native_message()

    async def scenario() -> dict[str, str] | None:
        event = _message_event(message=message)
        result: dict[str, str] | None = hook(
            event=event,
            gateway=object(),
            session_store=object(),
        )
        # The gateway drops the MessageEvent on skip: no LLM/core
        # continuation is possible from this hook result.
        assert result == {"action": "skip", "reason": "blackcat-finance"}
        await _drain_hook_tasks(hook)
        return result

    result = asyncio.run(scenario())
    assert result == {"action": "skip", "reason": "blackcat-finance"}
    assert len(calls) == 1
    assert calls[0].argv == [
        VALID_SETTINGS["cli_path"],
        "ingest-telegram",
        "--text",
        "+25 Работа Проект A",
        "--chat-id",
        str(CHAT_ID),
        "--thread-id",
        str(THREAD_ID),
        "--message-id",
        "555",
        "--update-id",
        "9001",
        "--transaction-date",
        "2026-09-04",
        "--received-at",
        "2026-09-04T18:30:00+00:00",
    ]
    assert message.replies == [RESPONSE_CREATED]
    assert _hook_tasks(hook) == set()


# ------------------------------------------------------------------
# Gateway recovery fallback: routing/provenance separation
# ------------------------------------------------------------------


def test_known_candidate_without_raw_message_is_consumed_not_passed_through(
    monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Missing raw_message is NOT unrelated traffic after routing.

    A KNOWN Finance candidate (normalized Telegram source + Finance
    chat/thread + candidate text) with ``raw_message=None`` must be
    consumed (skip, never the LLM) while the CLI is never invoked;
    only a safe value-free diagnostic is logged.
    """
    plugin, _, hook = _registered_hook(monkeypatch)
    calls: list[SpawnCall] = []
    _install_spawn_capture(monkeypatch, plugin, calls)
    event = _message_event(
        raw_message=None,
        text="+25 SECRET-TEXT-SENTINEL Проект A",
    )
    with caplog.at_level(logging.ERROR):
        result = hook(event=event, gateway=object(), session_store=object())
    # Consumed, not passed through.
    assert result == {"action": "skip", "reason": "blackcat-finance"}
    # No CLI, no task, no fabricated success.
    assert calls == []
    assert _hook_tasks(hook) == set()
    # Safe diagnostic only: no transaction text, no traceback.
    assert "SECRET-TEXT-SENTINEL" not in caplog.text
    assert "Traceback" not in caplog.text


# ------------------------------------------------------------------
# Gateway recovery fallback: result mapping
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("disposition", "expected_reply"),
    [
        ("CREATED", RESPONSE_CREATED),
        ("DUPLICATE_MESSAGE", RESPONSE_DUPLICATE),
        ("DUPLICATE_UPDATE", RESPONSE_DUPLICATE),
    ],
)
def test_fallback_preserves_the_fixed_replies_for_dispositions(
    monkeypatch: MonkeyPatch, disposition: str, expected_reply: str
) -> None:
    plugin, _, hook = _registered_hook(monkeypatch)
    stdout = json.dumps(
        {"disposition": disposition, "transaction_id": "t-1"}
    ).encode("utf-8")
    calls: list[SpawnCall] = []
    _install_spawn_capture(monkeypatch, plugin, calls, result=(0, stdout, b""))
    message = _native_message()

    async def scenario() -> None:
        assert hook(_message_event(message=message)) == {
            "action": "skip",
            "reason": "blackcat-finance",
        }
        await _drain_hook_tasks(hook)

    asyncio.run(scenario())
    assert len(calls) == 1
    assert message.replies == [expected_reply]


@pytest.mark.parametrize(
    "result",
    [
        (1, b"", b"Traceback (most recent call last):\n  File \"C:/secret/db.sqlite\""),
        (2, b"not json at all", b""),
        (0, b'{"disposition": "REJECTED"}', b""),
        (0, b'{"no_disposition": true}', b""),
    ],
)
def test_fallback_reply_is_the_generic_failure_for_every_failure_mode(
    monkeypatch: MonkeyPatch, result: tuple[int | None, bytes, bytes]
) -> None:
    plugin, _, hook = _registered_hook(monkeypatch)
    calls: list[SpawnCall] = []
    _install_spawn_capture(monkeypatch, plugin, calls, result=result)
    message = _native_message()

    async def scenario() -> None:
        assert hook(_message_event(message=message)) == {
            "action": "skip",
            "reason": "blackcat-finance",
        }
        await _drain_hook_tasks(hook)

    asyncio.run(scenario())
    assert len(calls) == 1
    assert message.replies == [RESPONSE_FAILURE]


def test_fallback_reply_is_the_generic_failure_when_the_cli_cannot_start(
    monkeypatch: MonkeyPatch,
) -> None:
    plugin, _, hook = _registered_hook(monkeypatch)
    calls: list[SpawnCall] = []
    _install_spawn_capture(monkeypatch, plugin, calls, error=OSError("spawn boom"))
    message = _native_message()

    async def scenario() -> None:
        assert hook(_message_event(message=message)) == {
            "action": "skip",
            "reason": "blackcat-finance",
        }
        await _drain_hook_tasks(hook)

    asyncio.run(scenario())
    # No automatic retry: the finance CLI is attempted exactly once.
    assert len(calls) == 1
    assert message.replies == [RESPONSE_FAILURE]


# ------------------------------------------------------------------
# Gateway recovery fallback: provenance
# ------------------------------------------------------------------


def test_fallback_cli_receives_exact_native_provenance(
    monkeypatch: MonkeyPatch,
) -> None:
    """Distinctive ids prove every value comes from its exact source."""
    settings = dict(VALID_SETTINGS, chat_id=-100222333444, thread_id=424)
    plugin, _, hook = _registered_hook(monkeypatch, settings)
    calls: list[SpawnCall] = []
    _install_spawn_capture(
        monkeypatch,
        plugin,
        calls,
        result=(0, b'{"disposition": "CREATED", "transaction_id": "t-1"}', b""),
    )
    message = _native_message(
        chat_id=-100222333444,
        thread_id=424,
        message_id=987654,
        text="-130.50 Инфраструктура Хостинг",
    )
    # The normalized source carries the same Finance chat/thread as
    # the custom settings, in canonical string form.
    source = FakeSessionSource(
        platform=FakePlatform("telegram"), chat_id="-100222333444", thread_id="424"
    )
    event = _message_event(
        message=message, source=source, platform_update_id=555001
    )

    async def scenario() -> None:
        assert hook(event) == {"action": "skip", "reason": "blackcat-finance"}
        await _drain_hook_tasks(hook)

    asyncio.run(scenario())
    assert len(calls) == 1
    argv = calls[0].argv
    # The update id comes from event.platform_update_id.
    assert argv[argv.index("--update-id") + 1] == "555001"
    # The chat/thread/message ids come from the native raw_message.
    assert argv[argv.index("--chat-id") + 1] == "-100222333444"
    assert argv[argv.index("--thread-id") + 1] == "424"
    assert argv[argv.index("--message-id") + 1] == "987654"
    # The exact message text is one single unchanged argv element.
    assert argv[argv.index("--text") + 1] == "-130.50 Инфраструктура Хостинг"


def test_fallback_timestamp_and_business_date_come_from_the_native_message(
    monkeypatch: MonkeyPatch,
) -> None:
    # 2026-09-04 23:30 UTC is already 2026-09-05 02:30 in Moscow.
    plugin, _, hook = _registered_hook(
        monkeypatch, dict(VALID_SETTINGS, business_timezone="Europe/Moscow")
    )
    calls: list[SpawnCall] = []
    _install_spawn_capture(
        monkeypatch,
        plugin,
        calls,
        result=(0, b'{"disposition": "CREATED", "transaction_id": "t-1"}', b""),
    )
    message = _native_message(date=datetime(2026, 9, 4, 23, 30, tzinfo=UTC))

    async def scenario() -> None:
        assert hook(_message_event(message=message)) == {
            "action": "skip",
            "reason": "blackcat-finance",
        }
        await _drain_hook_tasks(hook)

    asyncio.run(scenario())
    argv = calls[0].argv
    assert argv[argv.index("--transaction-date") + 1] == "2026-09-05"
    assert argv[argv.index("--received-at") + 1] == "2026-09-04T23:30:00+00:00"


def test_fallback_never_fabricates_a_wall_clock_timestamp(
    monkeypatch: MonkeyPatch,
) -> None:
    """A native timestamp far from today reaches the CLI verbatim."""
    plugin, _, hook = _registered_hook(monkeypatch)
    calls: list[SpawnCall] = []
    _install_spawn_capture(
        monkeypatch,
        plugin,
        calls,
        result=(0, b'{"disposition": "CREATED", "transaction_id": "t-1"}', b""),
    )
    message = _native_message(date=datetime(2020, 1, 2, 3, 4, tzinfo=UTC))

    async def scenario() -> None:
        assert hook(_message_event(message=message)) == {
            "action": "skip",
            "reason": "blackcat-finance",
        }
        await _drain_hook_tasks(hook)

    asyncio.run(scenario())
    argv = calls[0].argv
    assert argv[argv.index("--received-at") + 1] == "2020-01-02T03:04:00+00:00"
    assert argv[argv.index("--transaction-date") + 1] == "2020-01-02"


def _invalid_provenance_event(field: str, value: Any) -> FakeMessageEvent:
    """Build a routed Finance candidate whose ingestion provenance is invalid.

    Routing is always valid (normalized Telegram Finance source); only
    the raw native provenance named by ``field`` is broken.
    """
    if field == "platform_update_id":
        return _message_event(platform_update_id=value)
    message = _native_message()
    if field == "native_chat_id":
        # Normalized/native routing mismatch on the chat id.
        message.chat = FakeChat(value)
    elif field == "native_thread_id":
        # Normalized/native routing mismatch on the thread id.
        message.message_thread_id = value
    elif field == "date":
        # ``_native_message`` substitutes a default for date=None, so
        # a missing timestamp must be forced through the message.
        message.date = value
    else:
        message.message_id = value
    return _message_event(message=message)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("platform_update_id", None),
        ("platform_update_id", -1),
        ("platform_update_id", True),
        ("platform_update_id", "9001"),
        ("message_id", None),
        ("message_id", 0),
        ("message_id", -5),
        ("message_id", True),
        ("date", None),
        ("native_chat_id", -1009999999999),
        ("native_thread_id", 99),
    ],
)
def test_fallback_invalid_provenance_still_skips_and_never_invokes_the_cli(
    monkeypatch: MonkeyPatch, field: str, value: Any
) -> None:
    """Invalid raw provenance fails closed WITHOUT pass-through.

    The candidate is a KNOWN Finance candidate by normalized routing,
    so the hook always returns skip (never the LLM); the CLI is never
    invoked; the fixed failure reply is sent through the native
    message wherever one exists.
    """
    plugin, _, hook = _registered_hook(monkeypatch)
    calls: list[SpawnCall] = []
    _install_spawn_capture(
        monkeypatch,
        plugin,
        calls,
        result=(0, b'{"disposition": "CREATED", "transaction_id": "t-1"}', b""),
    )
    event = _invalid_provenance_event(field, value)
    message = event.raw_message

    async def scenario() -> None:
        # Skip FIRST: the candidate is consumed even though the CLI
        # will not run.
        assert hook(event=event, gateway=object()) == {
            "action": "skip",
            "reason": "blackcat-finance",
        }
        await _drain_hook_tasks(hook)

    asyncio.run(scenario())
    assert calls == []
    assert message.replies == [RESPONSE_FAILURE]
    assert _hook_tasks(hook) == set()


def test_fallback_naive_timestamp_never_invokes_the_cli(
    monkeypatch: MonkeyPatch,
) -> None:
    plugin, _, hook = _registered_hook(monkeypatch)
    calls: list[SpawnCall] = []
    _install_spawn_capture(monkeypatch, plugin, calls)
    message = _native_message(
        date=datetime(2026, 9, 4, 18, 30, tzinfo=UTC).replace(tzinfo=None)
    )
    event = _message_event(message=message)

    async def scenario() -> None:
        assert hook(event) == {"action": "skip", "reason": "blackcat-finance"}
        await _drain_hook_tasks(hook)

    asyncio.run(scenario())
    assert calls == []
    assert message.replies == [RESPONSE_FAILURE]


def test_fallback_inconsistent_event_text_never_invokes_the_cli(
    monkeypatch: MonkeyPatch,
) -> None:
    """The event text must match the native message text exactly."""
    plugin, _, hook = _registered_hook(monkeypatch)
    calls: list[SpawnCall] = []
    _install_spawn_capture(monkeypatch, plugin, calls)
    event = _message_event(text="+25 Работа Проект A (изменено)")

    async def scenario() -> None:
        assert hook(event) == {"action": "skip", "reason": "blackcat-finance"}
        await _drain_hook_tasks(hook)

    asyncio.run(scenario())
    assert calls == []
    assert event.raw_message.replies == [RESPONSE_FAILURE]


def test_fallback_missing_native_text_never_invokes_the_cli(
    monkeypatch: MonkeyPatch,
) -> None:
    plugin, _, hook = _registered_hook(monkeypatch)
    calls: list[SpawnCall] = []
    _install_spawn_capture(monkeypatch, plugin, calls)
    event = _message_event(message=_native_message(text=None), text="+25 Работа A")

    async def scenario() -> None:
        assert hook(event) == {"action": "skip", "reason": "blackcat-finance"}
        await _drain_hook_tasks(hook)

    asyncio.run(scenario())
    assert calls == []
    assert event.raw_message.replies == [RESPONSE_FAILURE]


# ------------------------------------------------------------------
# Gateway recovery fallback: task lifecycle
# ------------------------------------------------------------------


def test_fallback_tasks_are_strongly_retained_until_done(
    monkeypatch: MonkeyPatch,
) -> None:
    plugin, _, hook = _registered_hook(monkeypatch)
    release = asyncio.Event()

    async def blocking_spawn(
        argv: list[str], env: dict[str, str]
    ) -> tuple[int | None, bytes, bytes]:
        await release.wait()
        return 0, b'{"disposition": "CREATED", "transaction_id": "t-1"}', b""

    monkeypatch.setattr(plugin, "_spawn_process", blocking_spawn)

    async def scenario() -> None:
        for index in range(3):
            message = _native_message(message_id=700 + index)
            event = _message_event(message=message, platform_update_id=9200 + index)
            assert hook(event) == {"action": "skip", "reason": "blackcat-finance"}
        # Every in-flight task is strongly retained by the hook and
        # cannot disappear through garbage collection.
        for _ in range(100):
            if len(_hook_tasks(hook)) == 3:
                break
            await asyncio.sleep(0)
        assert len(_hook_tasks(hook)) == 3
        release.set()
        await _drain_hook_tasks(hook)

    asyncio.run(scenario())
    assert _hook_tasks(hook) == set()


def test_fallback_task_is_removed_after_completion(
    monkeypatch: MonkeyPatch,
) -> None:
    plugin, _, hook = _registered_hook(monkeypatch)
    calls: list[SpawnCall] = []
    _install_spawn_capture(
        monkeypatch,
        plugin,
        calls,
        result=(0, b'{"disposition": "CREATED", "transaction_id": "t-1"}', b""),
    )

    async def scenario() -> None:
        assert hook(_message_event()) == {
            "action": "skip",
            "reason": "blackcat-finance",
        }
        # Synchronously after scheduling, the task is retained...
        assert len(_hook_tasks(hook)) == 1
        await _drain_hook_tasks(hook)
        # ...and the done callback removed it after completion.
        assert _hook_tasks(hook) == set()

    asyncio.run(scenario())
    assert len(calls) == 1
    assert _hook_tasks(hook) == set()


def test_fallback_detached_exception_is_consumed_safely(
    monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A crashing fallback task never leaks its exception detail."""
    plugin, _, hook = _registered_hook(monkeypatch)

    async def exploding(config: Any, event: Any, raw_message: Any) -> None:
        raise RuntimeError("SECRET-FALLBACK-SENTINEL C:/secret/cli-path boom")

    monkeypatch.setattr(plugin, "_fallback_ingest", exploding)

    async def scenario() -> None:
        assert hook(_message_event()) == {
            "action": "skip",
            "reason": "blackcat-finance",
        }
        await _drain_hook_tasks(hook)

    with caplog.at_level(logging.ERROR):
        asyncio.run(scenario())
    log_text = caplog.text
    # The detached exception was consumed and only its TYPE is logged.
    assert "RuntimeError" in log_text
    assert "SECRET-FALLBACK-SENTINEL" not in log_text
    assert "cli-path" not in log_text
    assert "Traceback" not in log_text
    assert _hook_tasks(hook) == set()


def test_fallback_no_unbounded_task_accumulation(
    monkeypatch: MonkeyPatch,
) -> None:
    plugin, _, hook = _registered_hook(monkeypatch)
    calls: list[SpawnCall] = []
    _install_spawn_capture(
        monkeypatch,
        plugin,
        calls,
        result=(0, b'{"disposition": "CREATED", "transaction_id": "t-1"}', b""),
    )

    async def scenario() -> None:
        for index in range(5):
            message = _native_message(message_id=600 + index)
            event = _message_event(message=message, platform_update_id=9100 + index)
            assert hook(event) == {"action": "skip", "reason": "blackcat-finance"}
            await _drain_hook_tasks(hook)
            # Each completed task was cleaned up immediately.
            assert _hook_tasks(hook) == set()

    asyncio.run(scenario())
    assert len(calls) == 5


def test_fallback_scheduling_failure_fails_closed_against_the_llm(
    monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Without a running loop the candidate is still never LLM-routed."""
    plugin, _, hook = _registered_hook(monkeypatch)
    calls: list[SpawnCall] = []
    _install_spawn_capture(monkeypatch, plugin, calls)
    event = _message_event()
    with caplog.at_level(logging.ERROR):
        # Synchronous call outside any asyncio loop: scheduling itself
        # fails.
        result = hook(event)
    # Fail closed: the candidate is consumed regardless.
    assert result == {"action": "skip", "reason": "blackcat-finance"}
    # No fabricated success: nothing ran, nothing was replied.
    assert calls == []
    assert event.raw_message.replies == []
    # Safe diagnostic only: the exception type, no traceback.
    assert "RuntimeError" in caplog.text
    assert "Traceback" not in caplog.text


# ------------------------------------------------------------------
# Gateway recovery fallback: architecture
# ------------------------------------------------------------------


def test_manifest_declares_the_pre_gateway_dispatch_hook() -> None:
    manifest = (PLUGIN_INIT.parent / "plugin.yaml").read_text(encoding="utf-8")
    match = re.search(r"(?m)^hooks:\s*\n((?:[ \t]*-[ \t]*\S[^\n]*\n?)*)", manifest)
    assert match is not None
    declared = [
        line.strip()[1:].strip()
        for line in match.group(1).splitlines()
        if line.strip()
    ]
    assert declared == ["pre_gateway_dispatch"]


def test_plugin_imports_no_hermes_internals_at_any_depth() -> None:
    """No Hermes internal module is imported anywhere in the plugin.

    The only Hermes dependency of the plugin is the runtime ctx
    contract (``ctx.register_platform_handler`` /
    ``ctx.register_hook``) and the duck-typed hook payload fields;
    neither requires importing ``gateway.*``, ``hermes_cli.*``, or any
    other Hermes internal module.
    """
    tree = ast.parse(PLUGIN_INIT.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    for name in imported:
        assert not name.startswith("gateway"), name
        assert not name.startswith("hermes"), name
