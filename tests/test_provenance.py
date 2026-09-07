"""Tests for the Telegram provenance/idempotency contract (stage A2).

All tests are deterministic: no Telegram API, no network, no filesystem
access beyond normal package import, no wall-clock time, no randomness,
and no external services.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from hermes_finance import TelegramMessageRef

VALID_REF = TelegramMessageRef(
    chat_id=-1001234567890,
    message_thread_id=17,
    message_id=4242,
    update_id=9007199254740993,
)


def make_ref(**overrides: Any) -> TelegramMessageRef:
    """Build a valid TelegramMessageRef, applying keyword overrides (possibly invalid)."""
    params: dict[str, Any] = {
        "chat_id": -1001234567890,
        "message_thread_id": 17,
        "message_id": 4242,
        "update_id": 12345,
    }
    params.update(overrides)
    return TelegramMessageRef(**params)


class TestValidRefs:
    def test_negative_chat_id_accepted(self) -> None:
        ref = make_ref(chat_id=-1001234567890)
        assert ref.chat_id == -1001234567890

    def test_positive_chat_id_accepted(self) -> None:
        ref = make_ref(chat_id=987654321)
        assert ref.chat_id == 987654321

    def test_positive_message_thread_id_accepted(self) -> None:
        ref = make_ref(message_thread_id=1)
        assert ref.message_thread_id == 1

    def test_positive_message_id_accepted(self) -> None:
        ref = make_ref(message_id=1)
        assert ref.message_id == 1

    def test_update_id_zero_accepted(self) -> None:
        ref = make_ref(update_id=0)
        assert ref.update_id == 0

    def test_positive_update_id_accepted(self) -> None:
        ref = make_ref(update_id=9007199254740993)
        assert ref.update_id == 9007199254740993

    def test_all_fields_kept_exactly(self) -> None:
        ref = make_ref()
        assert ref.chat_id == -1001234567890
        assert ref.message_thread_id == 17
        assert ref.message_id == 4242
        assert ref.update_id == 12345


class TestMessageIdentity:
    def test_identity_is_stable_for_same_ref_values(self) -> None:
        first = make_ref()
        second = make_ref()
        assert first.message_identity == second.message_identity
        assert first.message_identity.key == second.message_identity.key

    def test_identity_key_is_deterministic_string(self) -> None:
        identity = make_ref(chat_id=-100, message_id=5).message_identity
        assert identity.key == "telegram-message:-100:5"
        assert isinstance(identity.key, str)

    def test_same_message_identity_despite_different_thread_id(self) -> None:
        base = make_ref(message_thread_id=17)
        other_topic = make_ref(message_thread_id=999)
        assert base.message_identity == other_topic.message_identity
        assert base.message_identity.key == other_topic.message_identity.key

    def test_same_message_identity_despite_different_update_id(self) -> None:
        base = make_ref(update_id=1)
        redelivered = make_ref(update_id=2)
        assert base.message_identity == redelivered.message_identity
        assert base.message_identity.key == redelivered.message_identity.key

    def test_same_message_identity_despite_different_thread_and_update_id(self) -> None:
        base = make_ref(message_thread_id=17, update_id=1)
        other = make_ref(message_thread_id=42, update_id=2)
        assert base.message_identity == other.message_identity
        assert base.message_identity.key == other.message_identity.key

    def test_differing_message_id_means_differing_identity(self) -> None:
        base = make_ref(message_id=4242)
        other = make_ref(message_id=4243)
        assert base.message_identity != other.message_identity
        assert base.message_identity.key != other.message_identity.key

    def test_differing_chat_id_means_differing_identity(self) -> None:
        base = make_ref(chat_id=-1001234567890)
        other = make_ref(chat_id=-1009876543210)
        assert base.message_identity != other.message_identity
        assert base.message_identity.key != other.message_identity.key

    def test_identity_equality_and_key_agree(self) -> None:
        a = make_ref().message_identity
        b = make_ref().message_identity
        assert (a == b) == (a.key == b.key)
        assert hash(a) == hash(b)


class TestUpdateIdentity:
    def test_update_identity_is_stable_for_same_update_id(self) -> None:
        first = make_ref(update_id=12345)
        second = make_ref(update_id=12345, message_thread_id=99)
        assert first.update_identity == second.update_identity
        assert first.update_identity.key == second.update_identity.key

    def test_update_identity_key_is_deterministic_string(self) -> None:
        identity = make_ref(update_id=7).update_identity
        assert identity.key == "telegram-update:7"
        assert isinstance(identity.key, str)

    def test_differing_update_id_means_differing_update_identity(self) -> None:
        base = make_ref(update_id=1)
        other = make_ref(update_id=2)
        assert base.update_identity != other.update_identity
        assert base.update_identity.key != other.update_identity.key

    def test_update_id_zero_has_valid_identity(self) -> None:
        identity = make_ref(update_id=0).update_identity
        assert identity.key == "telegram-update:0"


class TestImmutability:
    def test_ref_rejects_attribute_assignment(self) -> None:
        ref = make_ref()
        with pytest.raises(FrozenInstanceError):
            ref.chat_id = 1  # type: ignore[misc]

    def test_ref_rejects_thread_assignment(self) -> None:
        ref = make_ref()
        with pytest.raises(FrozenInstanceError):
            ref.message_thread_id = 2  # type: ignore[misc]

    def test_ref_rejects_update_id_assignment(self) -> None:
        ref = make_ref()
        with pytest.raises(FrozenInstanceError):
            ref.update_id = 3  # type: ignore[misc]

    def test_message_identity_rejects_attribute_assignment(self) -> None:
        identity = make_ref().message_identity
        with pytest.raises(FrozenInstanceError):
            identity.chat_id = 1  # type: ignore[misc]

    def test_update_identity_rejects_attribute_assignment(self) -> None:
        identity = make_ref().update_identity
        with pytest.raises(FrozenInstanceError):
            identity.update_id = 1  # type: ignore[misc]


class TestInvalidChatId:
    def test_bool_rejected(self) -> None:
        with pytest.raises(TypeError, match="chat_id must not be a bool"):
            make_ref(chat_id=True)

    def test_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="chat_id must not be zero"):
            make_ref(chat_id=0)

    def test_non_int_rejected(self) -> None:
        value: Any = "-1001234567890"
        with pytest.raises(TypeError, match="chat_id must be an int"):
            make_ref(chat_id=value)


class TestInvalidMessageThreadId:
    def test_bool_rejected(self) -> None:
        with pytest.raises(TypeError, match="message_thread_id must not be a bool"):
            make_ref(message_thread_id=False)

    def test_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="message_thread_id must be positive"):
            make_ref(message_thread_id=0)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="message_thread_id must be positive"):
            make_ref(message_thread_id=-1)

    def test_non_int_rejected(self) -> None:
        value: Any = 17.0
        with pytest.raises(TypeError, match="message_thread_id must be an int"):
            make_ref(message_thread_id=value)


class TestInvalidMessageId:
    def test_bool_rejected(self) -> None:
        with pytest.raises(TypeError, match="message_id must not be a bool"):
            make_ref(message_id=True)

    def test_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="message_id must be positive"):
            make_ref(message_id=0)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="message_id must be positive"):
            make_ref(message_id=-5)

    def test_non_int_rejected(self) -> None:
        value: Any = "4242"
        with pytest.raises(TypeError, match="message_id must be an int"):
            make_ref(message_id=value)


class TestInvalidUpdateId:
    def test_bool_rejected(self) -> None:
        with pytest.raises(TypeError, match="update_id must not be a bool"):
            make_ref(update_id=False)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="update_id must not be negative"):
            make_ref(update_id=-1)

    def test_non_int_rejected(self) -> None:
        value: Any = 12345.5
        with pytest.raises(TypeError, match="update_id must be an int"):
            make_ref(update_id=value)
