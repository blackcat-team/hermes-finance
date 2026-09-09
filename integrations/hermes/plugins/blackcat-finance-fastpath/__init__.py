# ruff: noqa: N999
# (The hyphenated directory name is the mandated Hermes user-plugin
# directory convention for this standalone plugin; the module itself is
# loaded by the Hermes host from the plugin directory, never imported
# as a regular Python package name.)
"""Standalone Hermes user plugin: blackcat-finance-fastpath (stage F5).

Tiny Telegram Finance fast-path for the Hermes host with TWO
independent delivery paths over one shared ingest core:

    Telegram Finance Topic
            |
    native platform handler   (PRIMARY fast path, unchanged semantics)
    pre_gateway_dispatch hook (FAIL-SAFE recovery path, Finance-side)
            | subprocess (argv only, never a shell)
            v
    hermes-finance ingest-telegram
            |
            v
    accepted finance core
            |
            v
    SQLite

PRIMARY PATH: the registered ``"telegram"`` platform-handler factory
wires a scoped python-telegram-bot MessageHandler into the adapter
Application. It stays the fastest path with the exact original
behavior.

FALLBACK PATH: Hermes is an independently updating host, and a
Telegram adapter reconnect/rebuild can temporarily lose the native
``ctx.register_platform_handler`` wiring. ``register`` therefore also
registers one synchronous ``pre_gateway_dispatch`` hook as a second,
independent Finance-owned recovery path. The real host invokes the
hook with a keyword payload (``event``, ``gateway``,
``session_store``, ...) whose public contract is additive, so the
callback accepts and ignores arbitrary extra host kwargs and fails
safely on a missing or malformed event. ROUTING is established from
the normalized MessageEvent ONLY: a Telegram platform on the
normalized source (enum-like ``source.platform`` with
``.value == "telegram"``, or the raw string as a compatibility
shape), the exact configured chat and thread on the normalized
source ids, and candidate text. Once routing identifies a KNOWN
Finance candidate it can NEVER reach ordinary Hermes/LLM routing:
the hook always returns ``{"action": "skip", "reason":
"blackcat-finance"}``. RAW NATIVE provenance (``event.raw_message``
and ``event.platform_update_id``) is a SECOND gate used only for
ingestion: with a usable native message the hook schedules a
plugin-owned async task on the currently running asyncio loop — the
synchronous hook itself never blocks on subprocess execution — and
the task reuses the SAME ingest/result/reply logic as the native
handler; without a usable native message, with invalid provenance,
or when the scheduling itself fails, the CLI is not invoked, a safe
value-free diagnostic is logged, the fixed failure reply is sent
only where a native reply object exists, and success is never
fabricated. This is a fail-safe only: Hermes reconnect itself is NOT
fixed; the contract is that Finance remains deterministic even if
the native handler wiring disappears.

The plugin owns ONLY: native Telegram event access, exact Finance
topic routing, transport provenance extraction, subprocess invocation,
and a tiny deterministic confirmation/error reply.

Boundaries:

- no Hermes internals: this module never imports any Hermes internal
  module (``gateway.*``, ``hermes_cli.*``); its only Hermes dependency
  is the runtime ``ctx`` registration contract and the duck-typed
  hook payload / event fields
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
  means NEITHER the Finance Telegram handler NOR the gateway fallback
  hook is registered, no defaults are guessed, and Hermes continues
  normally
- genuinely scoped routing: the native PTB filter matches only the
  exact configured chat id, the exact configured topic thread id,
  and text beginning with ``+<digit>`` or ``-<digit>``; the fallback
  hook routes on the SAME scope through the normalized event source
  (Telegram platform, exact chat, exact thread, candidate text) and
  accepts a normalized id only as the exact expected integer or its
  exact canonical decimal string. Every unrelated message or event
  is ordinary Hermes traffic and passes through untouched
- routing/provenance separation: the fallback hook decides Finance
  membership from the normalized event ONLY; the raw native message
  is never consulted for routing and is used only as the ingestion
  provenance gate, so a known Finance candidate is always consumed
  (skip) even when its native provenance is missing or invalid
- authoritative provenance: only real native Telegram values
  (update id, chat id, thread id, message id, text, message date)
  reach the CLI. The native path extracts them from the PTB
  update/message; the fallback path takes the update id from
  ``event.platform_update_id`` and the chat id, thread id, message
  id, text (after event/native consistency validation), and message
  date from the native ``event.raw_message``, each also consistent
  with the normalized routing. Nothing is fabricated, defaulted, or
  derived from the wall clock, and missing or invalid provenance
  (wrong type, bool, zero/negative ids, naive timestamp,
  inconsistent text or routing) means the CLI is not invoked at all;
  the candidate is still consumed (skip) and only the generic
  failure reply is sent through the native Telegram message where
  available
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
  stderr, argv, the CLI path, the database path, the child env,
  subprocess or task exception details, or Telegram payloads; a start
  failure logs only the exception type and a nonzero exit logs only
  the exit status
- non-blocking fallback hook: the synchronous ``pre_gateway_dispatch``
  hook never awaits and never runs the subprocess; all CLI work
  happens in the scheduled plugin-owned async task, and if that
  scheduling itself fails the candidate is still consumed (skip),
  a safe diagnostic is logged, and success is never fabricated
- mutual exclusion: the native PTB Finance handler stays PRIMARY —
  when it matches, ordinary Hermes normalized dispatch should not
  proceed to this fallback; the fallback exists for the observed
  condition where the native wiring is absent after a reconnect. The
  CLI-level (chat_id, message_id)/update_id deduplication is
  defense-in-depth only, never the normal-flow mutual-exclusion
  mechanism
- fallback task lifecycle: the hook factory owns closure-held strong
  references to every scheduled fallback task, cleaned up on
  completion, so tasks cannot disappear through garbage collection
  and cannot accumulate without bound; detached task exceptions are
  consumed and logged as the exception type only
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
    "GATEWAY_FALLBACK_HOOK",
    "HERMES_FINANCE_DB_PATH",
    "PLUGIN_NAME",
    "REGISTERED_PLATFORM",
    "RESPONSE_CREATED",
    "RESPONSE_DUPLICATE",
    "RESPONSE_FAILURE",
    "SKIP_REASON",
    "register",
]

logger = logging.getLogger(__name__)

#: Plugin identity, matching the plugin.yaml manifest.
PLUGIN_NAME: Final[str] = "blackcat-finance-fastpath"

#: The single platform this plugin registers a handler factory for.
REGISTERED_PLATFORM: Final[str] = "telegram"

#: The documented Hermes gateway hook used by the independent
#: Finance-side recovery path registered by ``register``.
GATEWAY_FALLBACK_HOOK: Final[str] = "pre_gateway_dispatch"

#: Skip reason returned to the gateway when the fallback hook consumes
#: a Finance candidate; the gateway then drops the MessageEvent before
#: auth/pairing/agent dispatch.
SKIP_REASON: Final[str] = "blackcat-finance"

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


async def _ingest_candidate(
    config: _FastPathConfig,
    *,
    update_id: Any,
    chat_id: Any,
    thread_id: Any,
    message_id: Any,
    text: Any,
    message_date: Any,
    message: Any,
) -> None:
    """Validate one candidate's provenance, ingest it, and reply.

    This is the SINGLE shared ingest/result/reply core used by BOTH
    the native platform handler and the gateway fallback task, so
    finance result semantics can never diverge between the paths.
    """
    # Authoritative native Telegram provenance only. Nothing is
    # fabricated, defaulted, or inferred. Every id is validated
    # fail-closed BEFORE any subprocess invocation: wrong type,
    # bool, or out-of-range id means the CLI is never called.
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
        await _ingest_candidate(
            config,
            update_id=getattr(update, "update_id", None),
            chat_id=getattr(message, "chat_id", None),
            thread_id=getattr(message, "message_thread_id", None),
            message_id=getattr(message, "message_id", None),
            text=getattr(message, "text", None),
            message_date=getattr(message, "date", None),
            message=message,
        )

    return handle


async def _fallback_ingest(
    config: _FastPathConfig, event: Any, raw_message: Any
) -> None:
    """Gateway-fallback ingest task body (plugin-owned async task).

    Extracts the fallback provenance from the normalized gateway event
    and the native Telegram message it carries, then reuses the SAME
    shared ingest/result/reply core as the native handler. Text is
    accepted only when the normalized event text and the native
    message text are exactly consistent, and the native chat/thread
    ids only when they are exactly consistent with the normalized
    routing that identified the candidate; any inconsistency fails
    the provenance gate (no CLI, failure reply only). Any unexpected
    exception is consumed here and logged as the exception type only,
    so the detached task can never leak transaction text, argv,
    paths, child stdout/stderr, Telegram payloads, or secrets.
    """
    try:
        event_text = getattr(event, "text", None)
        native_text = getattr(raw_message, "text", None)
        text = (
            event_text
            if isinstance(event_text, str) and event_text == native_text
            else None
        )
        # The native ids must be exactly consistent with the
        # normalized routing; anything else is invalid provenance.
        chat_id = getattr(raw_message, "chat_id", None)
        if not _valid_chat_id(chat_id) or chat_id != config.chat_id:
            chat_id = None
        thread_id = getattr(raw_message, "message_thread_id", None)
        if not _valid_positive_id(thread_id) or thread_id != config.thread_id:
            thread_id = None
        await _ingest_candidate(
            config,
            update_id=getattr(event, "platform_update_id", None),
            chat_id=chat_id,
            thread_id=thread_id,
            message_id=getattr(raw_message, "message_id", None),
            text=text,
            message_date=getattr(raw_message, "date", None),
            message=raw_message,
        )
    except Exception as error:  # noqa: BLE001 (fail-closed: never crash the host)
        # Exception TYPE only: the message, traceback, and any
        # transport detail stay out of the log.
        logger.error(
            "blackcat-finance-fastpath: the gateway fallback ingest task"
            " failed (exception type: %s)",
            type(error).__name__,
        )


def _source_platform_matches(platform: Any) -> bool:
    """Match the Telegram platform on the normalized event source.

    ``platform`` may be an enum-like object whose ``.value`` is the
    platform name (the real Hermes ``SessionSource`` shape) or, as a
    compatibility shape, the raw string ``"telegram"``. Anything else
    is unrelated traffic. No Hermes module is imported.
    """
    if isinstance(platform, str):
        return platform == REGISTERED_PLATFORM
    value = getattr(platform, "value", None)
    return isinstance(value, str) and value == REGISTERED_PLATFORM


def _routing_id_matches(expected: int, value: Any) -> bool:
    """Strict routing-id match for a normalized source id.

    The Finance configuration keeps integer Telegram ids while the
    normalized source carries string ids, so a value matches only
    when it is the exact expected integer or its exact canonical
    decimal string representation (``str(expected)``, including the
    leading ``-`` of a negative Telegram chat id). Booleans, floats,
    whitespace-padded or otherwise malformed strings, and unrelated
    objects never match; no loose numeric parsing is performed.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value == expected
    if isinstance(value, str):
        return value == str(expected)
    return False


def _make_fallback_hook(config: _FastPathConfig) -> Callable[..., Any]:
    """Build the synchronous ``pre_gateway_dispatch`` recovery hook.

    The real host invokes the hook with a keyword payload (``event``,
    ``gateway``, ``session_store``, ...) whose public contract is
    additive, so the callback accepts and ignores arbitrary extra
    host kwargs and fails safely (pass-through) on a missing or
    malformed event.

    ROUTING uses the normalized MessageEvent ONLY — Telegram
    platform, exact configured chat, exact configured thread, and
    candidate text — and never consults the raw native message. Once
    routing identifies a KNOWN Finance candidate, the hook ALWAYS
    returns ``{"action": "skip", "reason": "blackcat-finance"}`` so
    the candidate can never reach ordinary Hermes/LLM routing.

    RAW NATIVE provenance is a SECOND gate, used only for ingestion:
    with a usable native message the hook schedules a plugin-owned
    async task on the currently running loop (the synchronous hook
    itself never blocks on subprocess execution) that reuses the SAME
    ingest/result/reply logic as the native handler; without a usable
    native message, or when the scheduling itself fails, the CLI is
    not invoked, a safe value-free diagnostic is logged, and success
    is never fabricated — but the skip is still returned.

    The retained-task set is closure-owned state created here at
    registration time; there is no mutable adapter-specific or
    module-level state.
    """
    # Strong references to every scheduled fallback task. Tasks are
    # removed by the done callback, so nothing accumulates without
    # bound and no task can be garbage-collected mid-flight.
    tasks: set[asyncio.Task[None]] = set()

    def _task_done(task: asyncio.Task[None]) -> None:
        # Cleanup plus safe detached-exception consumption: retrieving
        # the exception also marks it retrieved for asyncio, and only
        # the exception TYPE is ever logged.
        tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "blackcat-finance-fastpath: a gateway fallback task ended"
                " with an exception (exception type: %s)",
                type(error).__name__,
            )

    def fallback_hook(event: Any = None, **_kwargs: Any) -> Any:
        # Missing or malformed event: it cannot be identified as
        # Finance traffic, so it passes through like any unrelated
        # event (getattr on attribute-less objects yields None).
        if event is None:
            return None
        # ROUTING (normalized MessageEvent data only).
        text = getattr(event, "text", None)
        if not isinstance(text, str) or _CANDIDATE_PREFIX.match(text) is None:
            return None
        source = getattr(event, "source", None)
        if not _source_platform_matches(getattr(source, "platform", None)):
            return None
        if not _routing_id_matches(config.chat_id, getattr(source, "chat_id", None)):
            return None
        if not _routing_id_matches(
            config.thread_id, getattr(source, "thread_id", None)
        ):
            return None

        # KNOWN FINANCE CANDIDATE: from here the event can NEVER
        # reach ordinary Hermes/LLM routing, whatever happens below.
        # Ingestion provenance gate: a usable native Telegram message.
        raw_message = getattr(event, "raw_message", None)
        if raw_message is None:
            logger.error(
                "blackcat-finance-fastpath: Finance gateway candidate"
                " without a native Telegram message; the finance CLI"
                " was not invoked"
            )
            return {"action": "skip", "reason": SKIP_REASON}
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(_fallback_ingest(config, event, raw_message))
        except Exception as error:  # noqa: BLE001 (fail-closed: never allow LLM)
            # Scheduling failed: still consume the candidate, log a
            # safe diagnostic (exception type only), and never
            # fabricate success.
            logger.error(
                "blackcat-finance-fastpath: scheduling the gateway fallback"
                " task failed (exception type: %s)",
                type(error).__name__,
            )
            return {"action": "skip", "reason": SKIP_REASON}
        tasks.add(task)
        task.add_done_callback(_task_done)
        return {"action": "skip", "reason": SKIP_REASON}

    return fallback_hook


def register(ctx: Any) -> None:
    """Register the Finance Telegram fast-path on the host context.

    Called once by the Hermes host at startup. Reads only the
    plugin-relative settings through ``ctx.get_config``. When every
    setting is valid, registers BOTH independent Finance paths:

    - the PRIMARY native fast path: exactly one platform handler
      factory for ``"telegram"``;
    - the FAIL-SAFE gateway recovery path: one synchronous
      ``pre_gateway_dispatch`` hook that consumes Finance candidates
      before gateway dispatch even if the native platform handler
      wiring disappears after a Telegram adapter reconnect/rebuild.

    When any setting is missing, blank, or invalid, registers
    neither path, logs an actionable warning, and returns normally so
    Hermes keeps running with ordinary routing.
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
    ctx.register_hook(GATEWAY_FALLBACK_HOOK, _make_fallback_hook(config))
