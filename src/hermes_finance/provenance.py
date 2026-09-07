"""Telegram provenance and idempotency identity contract for Hermes Finance.

This module defines the validated Telegram origin identities that the future
ledger persistence layer will depend on. It is deliberately free of any
Telegram library dependency, transport, parsing, or storage logic: only pure
Python stdlib value objects with construction-time validation.

Idempotency contract (to be enforced by future persistence)
-----------------------------------------------------------

A. One ledger transaction per *logical Telegram message identity*, derived
   from ``(chat_id, message_id)``. ``message_thread_id`` and ``update_id``
   are excluded: the same Telegram message must keep the same ledger-origin
   identity even if it is delivered in a different topic thread or via a
   different update.

B. One processed Telegram delivery per *update identity*, derived from
   ``update_id`` alone. Re-delivery of the same update must not be processed
   twice.

No database constraints, deduplication caches, repositories, or services are
implemented here; stage A2 defines the validated identities only.

The identity representations are deterministic, immutable strings derived
explicitly from the field values. Python's built-in :func:`hash` is never
used as a persisted identity because ``hash()`` randomisation makes it
unstable across processes and restarts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = [
    "TelegramMessageIdentity",
    "TelegramMessageRef",
    "TelegramUpdateIdentity",
]

_MESSAGE_IDENTITY_PREFIX: Final[str] = "telegram-message"
_UPDATE_IDENTITY_PREFIX: Final[str] = "telegram-update"


def _require_int(value: object, field: str) -> int:
    """Require a real ``int`` (``bool`` is explicitly rejected)."""
    if isinstance(value, bool):
        raise TypeError(f"{field} must not be a bool")
    if not isinstance(value, int):
        raise TypeError(f"{field} must be an int")
    return value


def _require_nonzero_int(value: object, field: str) -> int:
    """Require a non-zero ``int``; negative and positive values are accepted."""
    result = _require_int(value, field)
    if result == 0:
        raise ValueError(f"{field} must not be zero")
    return result


def _require_positive_int(value: object, field: str) -> int:
    """Require a strictly positive ``int``."""
    result = _require_int(value, field)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _require_non_negative_int(value: object, field: str) -> int:
    """Require a non-negative ``int`` (zero is accepted)."""
    result = _require_int(value, field)
    if result < 0:
        raise ValueError(f"{field} must not be negative")
    return result


@dataclass(frozen=True, slots=True)
class TelegramMessageIdentity:
    """Immutable logical identity of a Telegram message.

    Derived from ``chat_id`` and ``message_id`` only. The thread and the
    delivering update are deliberately excluded so that the same Telegram
    message keeps the same ledger-origin identity regardless of processing
    context or update delivery.

    ``key`` is a deterministic string suitable for persistence-level
    uniqueness checks; it is stable across processes and restarts.
    """

    chat_id: int
    message_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "chat_id", _require_nonzero_int(self.chat_id, "chat_id"))
        object.__setattr__(self, "message_id", _require_positive_int(self.message_id, "message_id"))

    @property
    def key(self) -> str:
        """Deterministic persisted-identity string for the logical message."""
        return f"{_MESSAGE_IDENTITY_PREFIX}:{self.chat_id}:{self.message_id}"


@dataclass(frozen=True, slots=True)
class TelegramUpdateIdentity:
    """Immutable identity of a single Telegram update delivery.

    Derived from ``update_id`` only. ``key`` is a deterministic string
    suitable for persistence-level delivery-deduplication checks; it is
    stable across processes and restarts.
    """

    update_id: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "update_id", _require_non_negative_int(self.update_id, "update_id")
        )

    @property
    def key(self) -> str:
        """Deterministic persisted-identity string for the update delivery."""
        return f"{_UPDATE_IDENTITY_PREFIX}:{self.update_id}"


@dataclass(frozen=True, slots=True)
class TelegramMessageRef:
    """Immutable Telegram message provenance reference.

    Carries exactly the provenance fields required for Finance Topic
    audit and idempotency:

    - ``chat_id``: non-zero ``int``; Telegram chat identifiers may
      legitimately be negative, so both signs are accepted.
    - ``message_thread_id``: strictly positive ``int``; Finance Topic
      provenance context (the actual topic ID is configured by later
      Hermes integration layers, never encoded here).
    - ``message_id``: strictly positive ``int``.
    - ``update_id``: non-negative ``int``; no assumption is made about
      current Telegram update-id sizes.

    ``message_thread_id`` is provenance/audit/routing context only and is
    deliberately not part of the logical message identity.

    Idempotency contract enabled by this value object (enforced later by
    persistence, not here):

    A. one ledger transaction per logical message identity
       ``(chat_id, message_id)`` -- see :attr:`message_identity`;
    B. one processed delivery per update identity ``update_id`` --
       see :attr:`update_identity`.
    """

    chat_id: int
    message_thread_id: int
    message_id: int
    update_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "chat_id", _require_nonzero_int(self.chat_id, "chat_id"))
        object.__setattr__(
            self,
            "message_thread_id",
            _require_positive_int(self.message_thread_id, "message_thread_id"),
        )
        object.__setattr__(self, "message_id", _require_positive_int(self.message_id, "message_id"))
        object.__setattr__(self, "update_id", _require_non_negative_int(self.update_id, "update_id"))

    @property
    def message_identity(self) -> TelegramMessageIdentity:
        """Logical Telegram message identity: ``(chat_id, message_id)``.

        Thread and update identifiers are excluded so that the identity is
        invariant across topic-thread and update-delivery changes.
        """
        return TelegramMessageIdentity(
            chat_id=self.chat_id,
            message_id=self.message_id,
        )

    @property
    def update_identity(self) -> TelegramUpdateIdentity:
        """Identity of the delivering Telegram update: ``update_id``."""
        return TelegramUpdateIdentity(update_id=self.update_id)
