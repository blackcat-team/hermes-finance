# ruff: noqa: N999
# (The hyphenated directory name is the mandated Hermes user-plugin
# directory convention for this standalone plugin; the module itself is
# loaded by the Hermes host from the plugin directory, never imported
# as a regular Python package name.)
"""Standalone Hermes user plugin: blackcat-finance-fastpath (stage F5).

Tiny Telegram Finance fast-path for the Hermes host:

    Telegram Finance Topic
            |
    standalone Hermes user plugin (this module)
            | subprocess (argv only, never a shell)
            v
    hermes-finance ingest-telegram
            |
            v
    accepted finance core
            |
            v
    SQLite

The plugin owns ONLY: native Telegram event access, exact Finance
topic routing, transport provenance extraction, subprocess invocation,
and a tiny deterministic confirmation/error reply.

Boundaries:

- no finance import: this module never imports the finance package,
  never opens SQLite, and never parses finance grammar; it only
  executes the standalone ``hermes-finance`` CLI as a child process,
  and it performs no sys.path manipulation
- no Telegram SDK import at module import or registration time: the
  python-telegram-bot classes are imported only inside the
  platform-handler factory, following the documented Hermes plugin
  contract, so ``register(ctx)`` works even when the SDK is absent
- fail-closed config: ``register`` reads only plugin-relative settings
  through ``ctx.get_config``; any missing, blank, or invalid setting
  means the Finance Telegram handler is NOT registered, no defaults
  are guessed, and Hermes continues normally
- genuinely scoped handler: a PTB ``filters.MessageFilter`` subclass
  matches only the exact configured chat id, the exact configured topic
  thread id, and text beginning with ``+<digit>`` or ``-<digit>``;
  every unrelated message falls through to ordinary Hermes routing
- authoritative provenance: only real native Telegram values
  (update id, chat id, thread id, message id, text, message date)
  reach the CLI; nothing is fabricated, defaulted, or derived from
  the wall clock, and missing or invalid provenance (wrong type,
  bool, zero/negative ids) means the CLI is not invoked at all
- validated result contract: the CLI success JSON must be a dict with
  a string disposition of CREATED / DUPLICATE_MESSAGE /
  DUPLICATE_UPDATE and a non-empty string transaction_id; anything
  else (wrong types, missing or malformed transaction_id, malformed
  JSON) yields only the generic failure reply
- argv subprocess only: ``asyncio.create_subprocess_exec`` with one
  argv value per argument; never a shell
- child environment: the child env is a copy of the process
  environment plus ``HERMES_FINANCE_DB_PATH``; the parent
  ``os.environ`` is never mutated
- log safety: local diagnostics never contain raw child stdout or
  stderr, argv, the CLI path, the database path, the child env, or
  subprocess exception details; a start failure logs only the
  exception type and a nonzero exit logs only the exit status
- no LLM involvement and no secrets in replies: Telegram answers are
  limited to three fixed strings
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

__all__ = [
    "HERMES_FINANCE_DB_PATH",
    "PLUGIN_NAME",
    "REGISTERED_PLATFORM",
    "RESPONSE_CREATED",
    "RESPONSE_DUPLICATE",
    "RESPONSE_FAILURE",
    "register",
]

logger = logging.getLogger(__name__)

#: Plugin identity, matching the plugin.yaml manifest.
PLUGIN_NAME: Final[str] = "blackcat-finance-fastpath"

#: The single platform this plugin registers a handler factory for.
REGISTERED_PLATFORM: Final[str] = "telegram"

#: Environment variable carrying the finance database path to the
#: child CLI. Set per child invocation only; never written to the
#: process environment.
HERMES_FINANCE_DB_PATH: Final[str] = "HERMES_FINANCE_DB_PATH"

#: Fixed Telegram replies. Nothing else is ever sent to the user.
RESPONSE_CREATED: Final[str] = "✅ Записано"
RESPONSE_DUPLICATE: Final[str] = "↩️ Уже записано"
RESPONSE_FAILURE: Final[str] = "⚠️ Не удалось записать операцию"

#: Transport candidate gate: text must begin with ``+`` or ``-``
#: immediately followed by a digit. This is intentionally NOT the
#: finance grammar; the accepted finance parser in the CLI remains
#: authoritative.
_CANDIDATE_PREFIX: Final[re.Pattern[str]] = re.compile(r"[+-]\d")

#: Recognized dispositions of the CLI ingest result. The disposition
#: must be a string member of this set; anything else is a failure.
_RECOGNIZED_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {"CREATED", "DUPLICATE_MESSAGE", "DUPLICATE_UPDATE"}
)


class _ConfigError(ValueError):
    """One plugin setting is missing, blank, or invalid."""


@dataclass(frozen=True)
class _FastPathConfig:
    """Validated plugin settings; no defaults exist."""

    cli_path: str
    database_path: str
    chat_id: int
    thread_id: int
    business_timezone: ZoneInfo


def _required_nonempty_str(key: str, value: Any) -> str:
    """Validate one required non-empty string setting."""
    if not isinstance(value, str) or not value.strip():
        raise _ConfigError(f"plugin setting {key!r} must be a non-empty string")
    return value.strip()


def _required_chat_id(value: Any) -> int:
    """Validate the exact Telegram chat id: real int, bool rejected, non-zero."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise _ConfigError(
            "plugin setting 'chat_id' must be the exact Telegram chat id as an"
            " integer (for example -1001234567890)"
        )
    if value == 0:
        raise _ConfigError("plugin setting 'chat_id' must be a non-zero Telegram chat id")
    return value


def _required_thread_id(value: Any) -> int:
    """Validate the exact Telegram topic id: real int, bool rejected, > 0."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise _ConfigError(
            "plugin setting 'thread_id' must be the exact Telegram Finance topic"
            " id (message_thread_id) as an integer"
        )
    if value <= 0:
        raise _ConfigError("plugin setting 'thread_id' must be a positive topic id")
    return value


def _required_timezone(value: Any) -> ZoneInfo:
    """Validate one required non-empty IANA timezone identifier."""
    name = _required_nonempty_str("business_timezone", value)
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise _ConfigError(
            "plugin setting 'business_timezone' must be a valid IANA timezone"
            f" identifier, got {name!r}"
        ) from error


def _load_config(ctx: Any) -> _FastPathConfig:
    """Read and validate only the plugin-relative settings from ``ctx``."""
    return _FastPathConfig(
        cli_path=_required_nonempty_str("cli_path", ctx.get_config("cli_path")),
        database_path=_required_nonempty_str(
            "database_path", ctx.get_config("database_path")
        ),
        chat_id=_required_chat_id(ctx.get_config("chat_id")),
        thread_id=_required_thread_id(ctx.get_config("thread_id")),
        business_timezone=_required_timezone(ctx.get_config("business_timezone")),
    )


def _make_platform_factory(
    config: _FastPathConfig,
) -> Callable[[Any, Any], None]:
    """Build the native platform-handler factory for ``register``.

    The factory runs at connect time with the native
    python-telegram-bot ``application`` and a read-only ``adapter``.
    The Telegram SDK classes are imported only here, never at module
    import or registration time.
    """

    def factory(application: Any, adapter: Any) -> None:
        # Imported only inside the factory, per the documented Hermes
        # plugin platform-handler contract. The custom filter derives
        # from ``filters.MessageFilter``: the PTB runtime used by the
        # Hermes host does not export a top-level
        # ``telegram.ext.MessageFilter``.
        from telegram.ext import MessageHandler, filters

        class _FinanceTopicFilter(filters.MessageFilter):
            """Scoped match: exact chat, exact topic, candidate prefix.

            Only a normal text message in the exact configured Finance
            chat and topic thread whose text begins with ``+<digit>``
            or ``-<digit>`` matches. Every other update is left for
            ordinary Hermes core routing.
            """

            def filter(self, message: Any) -> bool:
                chat_id = getattr(message, "chat_id", None)
                if isinstance(chat_id, bool) or not isinstance(chat_id, int):
                    return False
                if chat_id != config.chat_id:
                    return False
                if getattr(message, "message_thread_id", None) != config.thread_id:
                    return False
                text = getattr(message, "text", None)
                if not isinstance(text, str):
                    return False
                return _CANDIDATE_PREFIX.match(text) is not None

        # Genuinely scoped handler: unrelated messages never select
        # this handler, so they stay eligible for Hermes core routing.
        application.add_handler(
            MessageHandler(_FinanceTopicFilter(), _make_callback(config))
        )

    return factory


async def _spawn_process(
    argv: list[str], env: dict[str, str]
) -> tuple[int | None, bytes, bytes]:
    """Run the finance CLI argv and capture stdout/stderr.

    Uses ``asyncio.create_subprocess_exec`` with one argv value per
    argument; never a shell. The child environment is the supplied
    per-invocation mapping; the process environment is untouched.
    """
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    stdout, stderr = await process.communicate()
    return process.returncode, stdout, stderr


async def _invoke_cli(argv: list[str], env: dict[str, str]) -> str:
    """Run the CLI once and map the outcome to a fixed reply string.

    Never retries and never claims success on a start failure, a
    nonzero exit, malformed output, an invalid result contract, or an
    unexpected disposition. Diagnostics are log-safe: no raw child
    stdout/stderr, argv, CLI path, database path, child env, or
    subprocess exception detail is ever logged; a start failure logs
    only the exception type and a nonzero exit logs only the exit
    status.
    """
    try:
        returncode, stdout, _stderr = await _spawn_process(argv, env)
    except Exception as error:  # noqa: BLE001 (fail-closed: never crash the host)
        # Exception TYPE only: the message, traceback, and any
        # transport detail stay out of the log. Child stderr is also
        # deliberately never logged anywhere.
        logger.error(
            "blackcat-finance-fastpath: failed to start the finance CLI"
            " (exception type: %s)",
            type(error).__name__,
        )
        return RESPONSE_FAILURE
    if returncode != 0:
        # Exit status only: raw stderr is never logged.
        logger.error(
            "blackcat-finance-fastpath: finance CLI failed with exit status %r",
            returncode,
        )
        return RESPONSE_FAILURE
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        # Raw child stdout is never logged.
        logger.error(
            "blackcat-finance-fastpath: finance CLI output was not valid JSON"
        )
        return RESPONSE_FAILURE
    if not isinstance(payload, dict):
        logger.error(
            "blackcat-finance-fastpath: finance CLI output was not a JSON object"
        )
        return RESPONSE_FAILURE
    disposition = payload.get("disposition")
    transaction_id = payload.get("transaction_id")
    # The disposition TYPE is validated before the set membership
    # check so that a list or object disposition can never raise.
    if not isinstance(disposition, str) or disposition not in _RECOGNIZED_DISPOSITIONS:
        # The unexpected disposition value itself is never logged.
        logger.error(
            "blackcat-finance-fastpath: unexpected finance CLI disposition"
            " or disposition type"
        )
        return RESPONSE_FAILURE
    if not isinstance(transaction_id, str) or not transaction_id.strip():
        logger.error(
            "blackcat-finance-fastpath: finance CLI result lacks a valid"
            " transaction_id"
        )
        return RESPONSE_FAILURE
    if disposition == "CREATED":
        return RESPONSE_CREATED
    return RESPONSE_DUPLICATE


async def _reply(message: Any, text: str) -> None:
    """Best-effort tiny Telegram reply; never raises to the host."""
    try:
        await message.reply_text(text)
    except Exception as error:  # noqa: BLE001 (fail-closed: never crash the host)
        # Exception TYPE only: no traceback, no API detail.
        logger.error(
            "blackcat-finance-fastpath: failed to send the Telegram reply"
            " (exception type: %s)",
            type(error).__name__,
        )


def _valid_chat_id(value: Any) -> bool:
    """A native Telegram chat id: real int, bool rejected, non-zero."""
    return isinstance(value, int) and not isinstance(value, bool) and value != 0


def _valid_positive_id(value: Any) -> bool:
    """A native Telegram topic/message id: real int, bool rejected, > 0."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _valid_update_id(value: Any) -> bool:
    """A native Telegram update id: real int, bool rejected, >= 0."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _make_callback(config: _FastPathConfig) -> Callable[[Any, Any], Awaitable[None]]:
    """Build the scoped-message callback that invokes the finance CLI."""

    async def handle(update: Any, context: Any) -> None:
        message = getattr(update, "message", None)
        if message is None:
            logger.error(
                "blackcat-finance-fastpath: update without a message; the"
                " finance CLI was not invoked"
            )
            return

        # Authoritative native Telegram provenance only. Nothing is
        # fabricated, defaulted, or inferred. Every id is validated
        # fail-closed BEFORE any subprocess invocation: wrong type,
        # bool, or out-of-range id means the CLI is never called.
        update_id = getattr(update, "update_id", None)
        chat_id = getattr(message, "chat_id", None)
        thread_id = getattr(message, "message_thread_id", None)
        message_id = getattr(message, "message_id", None)
        text = getattr(message, "text", None)
        message_date = getattr(message, "date", None)

        date_valid = (
            isinstance(message_date, datetime)
            and message_date.tzinfo is not None
            and message_date.utcoffset() is not None
        )
        if (
            not _valid_chat_id(chat_id)
            or not _valid_positive_id(thread_id)
            or not _valid_positive_id(message_id)
            or not _valid_update_id(update_id)
            or not isinstance(text, str)
            or not text
            or not date_valid
        ):
            # Value-free diagnostic: no message text, timestamps, or
            # other message content is ever logged.
            logger.error(
                "blackcat-finance-fastpath: required native Telegram"
                " provenance is missing or invalid; the finance CLI was"
                " not invoked"
            )
            await _reply(message, RESPONSE_FAILURE)
            return

        # Business date from the native Telegram timestamp in the
        # configured business timezone; the native aware timestamp is
        # also the deterministic transport timestamp. No wall clock.
        transaction_date = message_date.astimezone(config.business_timezone).date()

        argv = [
            config.cli_path,
            "ingest-telegram",
            "--text",
            text,
            "--chat-id",
            str(chat_id),
            "--thread-id",
            str(thread_id),
            "--message-id",
            str(message_id),
            "--update-id",
            str(update_id),
            "--transaction-date",
            transaction_date.isoformat(),
            "--received-at",
            message_date.isoformat(),
        ]

        # Per-child environment only; the parent environment is never
        # mutated.
        child_env = dict(os.environ)
        child_env[HERMES_FINANCE_DB_PATH] = config.database_path

        reply_text = await _invoke_cli(argv, child_env)
        await _reply(message, reply_text)

    return handle


def register(ctx: Any) -> None:
    """Register the Finance Telegram fast-path on the host context.

    Called once by the Hermes host at startup. Reads only the
    plugin-relative settings through ``ctx.get_config``. When every
    setting is valid, registers exactly one platform handler factory
    for ``"telegram"``. When any setting is missing, blank, or
    invalid, registers nothing, logs an actionable warning, and
    returns normally so Hermes keeps running with ordinary routing.
    """
    try:
        config = _load_config(ctx)
    except _ConfigError as error:
        logger.warning(
            "blackcat-finance-fastpath: the Finance Telegram fast-path is"
            " DISABLED; fix the plugin settings to enable it: %s",
            error,
        )
        return
    ctx.register_platform_handler(REGISTERED_PLATFORM, _make_platform_factory(config))
