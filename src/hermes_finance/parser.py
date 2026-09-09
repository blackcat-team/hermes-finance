"""Deterministic fast-entry input parser for Hermes Finance (stage B1).

This module converts a strict user text command into a validated, immutable
:class:`ParsedTransactionInput`. It is intentionally free of Telegram
integration, timestamps, identifiers and persistence: those concerns belong
to later layers that will consume :class:`ParsedTransactionInput`.

Supported grammar
-----------------

::

    <sign><amount>[USDT] <category> <source> [| <comment>]
    <sign><amount> [USDT] <category> <source> [| <comment>]

- ``sign``: ``+`` for income, ``-`` for expense. The sign carries direction
  only; the amount handed to the domain is unsigned and must be positive.
- ``amount``: a clean decimal string (digits with an optional fraction).
  Exponent notation, thousands separators, commas and currency symbols are
  rejected. Full ``Decimal`` precision is preserved via
  :func:`hermes_finance.domain.normalize_usdt_amount`.
- ``USDT``: the only recognized optional currency token, either glued
  directly to the amount or separated by whitespace, matched ASCII
  case-insensitively via the explicitly ASCII-bounded pattern
  ``[Uu][Ss][Dd][Tt]``. Only ASCII spellings of U/S/D/T are accepted:
  Unicode case-conversion lookalikes such as ``ſ`` (U+017F LATIN SMALL
  LETTER LONG S, which ``str.upper`` maps to ``S``) are rejected as
  unsupported currency lookalikes instead of being reinterpreted as a
  category. No other currency token is ever guessed. Non-USDT text
  GLUED directly to the amount is malformed and rejected, because
  category syntax is impossible in that position (``-21BTC ChatGPT``).
  A SEPARATED token after the amount and the optional separated USDT
  is always the category, even when it is an uppercase ASCII token such
  as ``AI``, ``VPN`` or ``BTC``: the grammar is ambiguous between a
  currency and a category in that position, and category semantics win.
- ``category``: the first whitespace-separated token after the amount
  (and the optional USDT token). Uppercase ASCII categories such as
  ``AI``, ``API``, ``VPN``, ``VDS``, ``BTC`` and ``ETH`` are valid.
- ``source``: all remaining tokens before the optional comment delimiter,
  joined with single spaces.
- ``|``: optional comment delimiter. Everything after the *first* ``|`` is
  comment text: surrounding whitespace is trimmed, a blank comment
  normalises to ``None``, and further ``|`` characters inside the comment
  are preserved verbatim.

Whitespace behaviour
--------------------

Surrounding whitespace around the whole input is ignored. Structural
separators between fields may be any run of whitespace characters. As a
deterministic normalisation, runs of whitespace between source tokens are
collapsed to single spaces inside the parsed multi-word source; no other
text rewriting is performed.

Malformed input raises :class:`TransactionParseError` (a ``ValueError``
subclass); arbitrary internal exceptions are never leaked as part of the
public parse contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from hermes_finance.domain import (
    Direction,
    normalize_optional_text,
    normalize_required_text,
    normalize_usdt_amount,
)

__all__ = [
    "ParsedTransactionInput",
    "TransactionParseError",
    "parse_transaction_input",
]

_USDT_TOKEN: Final[str] = "USDT"

# ASCII-only case-insensitive USDT recognition. The character classes are
# deliberately bounded to ASCII letters; general Unicode .upper()/.lower()/
# .casefold() comparisons are never used for currency recognition because
# Unicode case conversion maps some non-ASCII characters onto ASCII letters
# (e.g. "ſ" U+017F upper-cases to "S").
_USDT_ASCII_RE: Final[re.Pattern[str]] = re.compile(r"[Uu][Ss][Dd][Tt]")

# Clean unsigned decimal: digits with an optional fractional part.
# Exponent notation, signs, separators and stray characters are rejected.
_AMOUNT_RE: Final[re.Pattern[str]] = re.compile(r"[0-9]+(?:\.[0-9]+)?")

# Bounded, deterministic unsupported-currency heuristic: an all-uppercase
# ASCII token of 2 to 6 letters glued directly to the amount (no
# whitespace separator) that is not the supported USDT token. No currency
# database is involved. This heuristic is applied ONLY in the glued
# position, where category syntax is impossible; a separated token with
# the same shape is a valid category (AI, VPN, BTC, ...), so the
# heuristic is never applied there.
_CURRENCY_LIKE_RE: Final[re.Pattern[str]] = re.compile(r"[A-Z]{2,6}")

_SIGN_TO_DIRECTION: Final[dict[str, Direction]] = {
    "+": Direction.INCOME,
    "-": Direction.EXPENSE,
}

_COMMENT_DELIMITER: Final[str] = "|"


def _is_ascii_usdt(token: str) -> bool:
    """True only for an exact ASCII case-insensitive spelling of USDT."""
    return _USDT_ASCII_RE.fullmatch(token) is not None


def _is_non_ascii_usdt_lookalike(token: str) -> bool:
    """True for non-ASCII tokens that Unicode case conversion maps to ``USDT``.

    ``str.upper()`` maps some non-ASCII characters onto ASCII letters (for
    example ``ſ`` U+017F LATIN SMALL LETTER LONG S becomes ``S``), so a
    naive ``token.upper() == "USDT"`` comparison would wrongly accept
    ``"uſdt"`` as the supported currency. Tokens containing any non-ASCII
    character can therefore never be the USDT token; tokens whose Unicode
    upper-case form equals ``USDT`` are rejected outright as unsupported
    currency lookalikes instead of being reinterpreted as a category.
    """
    return not token.isascii() and token.upper() == _USDT_TOKEN


class TransactionParseError(ValueError):
    """Raised when fast-entry text does not conform to the input grammar."""


@dataclass(frozen=True, slots=True)
class ParsedTransactionInput:
    """Immutable, validated structured finance input parsed from text.

    This is *not* a persisted :class:`hermes_finance.domain.Transaction`:
    ``transaction_date``, timestamps, persistence identifiers and Telegram
    provenance belong to later layers. Values are normalised with the
    accepted domain rules at construction time, so a frozen instance can
    never hold invalid money or text.
    """

    direction: Direction
    amount_usdt: Decimal
    category: str
    source: str
    comment: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.direction, Direction):
            raise TransactionParseError("direction must be a Direction member")
        try:
            object.__setattr__(self, "amount_usdt", normalize_usdt_amount(self.amount_usdt))
            object.__setattr__(self, "category", normalize_required_text(self.category, "category"))
            object.__setattr__(self, "source", normalize_required_text(self.source, "source"))
            object.__setattr__(self, "comment", normalize_optional_text(self.comment, "comment"))
        except (TypeError, ValueError) as exc:
            raise TransactionParseError(str(exc)) from exc


def parse_transaction_input(text: str) -> ParsedTransactionInput:
    """Parse strict fast-entry text into a :class:`ParsedTransactionInput`.

    See the module docstring for the grammar. Raises
    :class:`TransactionParseError` for any malformed input.
    """
    if not isinstance(text, str):
        raise TransactionParseError(f"transaction input must be a string, got {type(text).__name__!r}")

    raw = text.strip()
    if not raw:
        raise TransactionParseError("transaction input must not be blank")

    sign = raw[0]
    direction = _SIGN_TO_DIRECTION.get(sign)
    if direction is None:
        raise TransactionParseError("transaction input must start with '+' or '-'")

    amount_match = _AMOUNT_RE.match(raw, 1)
    if amount_match is None:
        if len(raw) > 1 and raw[1].isspace():
            raise TransactionParseError("whitespace is not allowed between the sign and the amount")
        raise TransactionParseError("a decimal amount is required immediately after the sign")
    amount_text = amount_match.group(0)
    rest = raw[amount_match.end() :]

    # Optional USDT token glued directly to the amount, e.g. "-1USDT".
    # Recognition is ASCII case-insensitive only. Any other non-whitespace
    # text glued to the amount is malformed; an all-uppercase ASCII 2-6
    # letter token there is an unsupported currency, as is any non-ASCII
    # Unicode lookalike of USDT.
    if rest and not rest[0].isspace():
        glued = rest.split(maxsplit=1)[0]
        if _is_ascii_usdt(glued):
            rest = rest[len(glued) :]
        elif _CURRENCY_LIKE_RE.fullmatch(glued) or _is_non_ascii_usdt_lookalike(glued):
            raise TransactionParseError(f"unsupported currency token {glued!r}; only USDT is accepted")
        else:
            raise TransactionParseError(f"malformed text glued to the amount: {glued!r}")

    body = rest.strip()
    head, _, comment_text = body.partition(_COMMENT_DELIMITER)
    comment = normalize_optional_text(comment_text, "comment")

    tokens = head.split()
    if tokens and _is_ascii_usdt(tokens[0]):
        tokens = tokens[1:]
    elif tokens and _is_non_ascii_usdt_lookalike(tokens[0]):
        # Unicode case-conversion lookalikes of the USDT token are never
        # accepted as the currency and never silently reinterpreted as
        # a category. Every other separated token — including an
        # uppercase ASCII token such as "AI", "VPN" or "BTC" — is the
        # category: no currency guessing happens in this position.
        raise TransactionParseError(
            f"unsupported currency token {tokens[0]!r}; only USDT is accepted"
        )

    if not tokens:
        raise TransactionParseError("category and source are missing after the amount")

    category = tokens[0]
    source_tokens = tokens[1:]
    if not source_tokens:
        raise TransactionParseError(f"source is missing after the category {category!r}")
    source = " ".join(source_tokens)

    try:
        amount = normalize_usdt_amount(amount_text)
    except ValueError as exc:
        raise TransactionParseError(str(exc)) from exc

    return ParsedTransactionInput(
        direction=direction,
        amount_usdt=amount,
        category=category,
        source=source,
        comment=comment,
    )
