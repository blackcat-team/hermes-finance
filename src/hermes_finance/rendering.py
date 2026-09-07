"""Deterministic plain-text rendering of finance views (stages E2, G1, G2).

This module is the presentation-only half of the reporting and listing
stages: it receives already-built, immutable result objects produced by
accepted read-only lower layers and projects them into one deterministic
human-readable Russian plain-text view suitable for later delivery by
Hermes.

Three view families exist:

- the stage-E2 monthly report:
  :func:`render_monthly_report` projects an immutable
  :class:`~hermes_finance.reporting.MonthlyFinanceReport` produced by
  the accepted E1 operation
  :func:`~hermes_finance.reporting.build_monthly_report`
- the stage-G1 transaction lists:
  :func:`render_recent_transactions`,
  :func:`render_transactions_by_date`, and
  :func:`render_transactions_by_month` project the immutable
  :class:`hermes_finance.domain.Transaction` tuples produced by the
  accepted D1 operations
- the stage-G2 filtered monthly summaries:
  :func:`render_monthly_category_summary` and
  :func:`render_monthly_source_summary` project one authoritative
  :class:`~hermes_finance.reporting.CategoryReport` /
  :class:`~hermes_finance.reporting.SourceReport` selected from an E1
  report by the accepted G2 selection layer, or the explicit "no data"
  state of a missing label

Contract highlights:

- rendering owns presentation only and never recalculates: every
  displayed value is taken exactly as stored on the input objects at
  the moment of rendering. No field is ever recomputed, re-summed,
  cross-checked, or "fixed"; report invariants belong entirely to E1
  and selection/ordering invariants belong entirely to D1.
- inputs must be the real accepted types; lookalike objects are
  rejected with :class:`TypeError` without duck-typing. Frozen inputs
  are never mutated: rendering is a pure projection.
- ordering is inherited, never re-derived: report categories render in
  the report's own tuple order, report sources in each category's
  tuple order, and transaction cards render in the exact sequence the
  D1 operation returned. This module adds no second sorting policy.
- month names are hard-coded deterministic uppercase Russian strings;
  there is no system-locale dependency, no ``strftime``, and no
  wall-clock access (dates, years, and months come from the inputs
  only).
- money display is context-independent and exact for every finite
  :class:`decimal.Decimal`: the canonical fixed-point form is built
  directly from ``Decimal.as_tuple()`` using string/integer placement
  only. There is no float conversion, no rounding, no quantisation, no
  scientific notation, no thousands separators, and no locale
  formatting, and the ambient decimal context is neither consulted nor
  mutated. Redundant trailing fractional zeros are removed for human
  readability and a mathematically integral value renders without a
  decimal point.
- transaction cards display the stable persisted
  ``transaction_id`` exactly as stored (never converted to an integer,
  never invented): a :class:`~hermes_finance.domain.Transaction` that
  reached this user-facing renderer without a real persisted id fails
  deterministically with :class:`ValueError` instead of showing a fake
  identifier.
- category, source, and comment labels are persisted business strings
  rendered verbatim except for a minimal visible-escape policy that
  keeps embedded control characters from breaking the layout:
  backslash becomes ``\\\\``, CR/LF/TAB become ``\\r``/``\\n``/``\\t``,
  and every other C0 control (U+0000..U+001F) plus DEL (U+007F)
  becomes a deterministic ``\\uXXXX`` escape with uppercase hexadecimal
  digits. Normal Cyrillic, Latin, spaces, and punctuation are
  untouched: no stripping, lowercasing, casefolding, HTML, or Markdown
  escaping.
- the output is plain text only: no Markdown, no HTML, no transport
  parse-mode syntax, no truncation, chunking, or size splitting. Every
  complete view is one string with no trailing whitespace and no final
  newline.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from typing import Final

from hermes_finance.domain import (
    Direction,
    Transaction,
    normalize_required_text,
    require_calendar_date,
)
from hermes_finance.reporting import (
    CategoryReport,
    MonthlyFinanceReport,
    SourceReport,
)

__all__ = [
    "render_monthly_category_summary",
    "render_monthly_report",
    "render_monthly_source_summary",
    "render_recent_transactions",
    "render_transactions_by_date",
    "render_transactions_by_month",
]

#: Deterministic hard-coded uppercase Russian month names, indexed by
#: month number. The values are stored pre-uppercased so the header is
#: independent of any casing transformation or locale machinery.
_MONTH_NAMES: Final[dict[int, str]] = {
    1: "ЯНВАРЬ",
    2: "ФЕВРАЛЬ",
    3: "МАРТ",
    4: "АПРЕЛЬ",
    5: "МАЙ",
    6: "ИЮНЬ",
    7: "ИЮЛЬ",
    8: "АВГУСТ",
    9: "СЕНТЯБРЬ",
    10: "ОКТЯБРЬ",
    11: "НОЯБРЬ",
    12: "ДЕКАБРЬ",
}


def _format_decimal(value: Decimal) -> str:
    """Render one finite Decimal as exact canonical fixed-point text.

    The digits and exponent are taken directly from ``as_tuple()`` and
    placed with string operations only, so the result is exact for
    arbitrary-precision values, independent of the ambient decimal
    context, and never in scientific notation. Redundant trailing
    fractional zeros are removed, and a mathematically integral value
    (including any signed zero) renders without a fraction. Non-finite
    values can never come from an accepted E1 report and are rejected.
    """
    sign, digits_tuple, exponent = value.as_tuple()
    if not isinstance(exponent, int):
        raise TypeError(f"exact rendering requires a finite Decimal, got {value!r}")
    digits = "".join(str(digit) for digit in digits_tuple)
    if not digits:
        digits = "0"
    # An exact zero (including a signed zero) always renders as "0".
    if digits.strip("0") == "":
        return "0"
    if exponent >= 0:
        magnitude = digits + "0" * exponent
        fraction = ""
    else:
        places = -exponent
        if len(digits) <= places:
            magnitude = "0"
            fraction = "0" * (places - len(digits)) + digits
        else:
            magnitude = digits[:-places]
            fraction = digits[-places:]
    fraction = fraction.rstrip("0")
    body = magnitude + ("." + fraction if fraction else "")
    return ("-" if sign else "") + body


def _format_net(value: Decimal) -> str:
    """Render a net total with the explicit-sign policy.

    Positive non-zero nets carry a leading ``+``, negative nets carry
    their ``-``, and an exact zero renders as ``0`` (never ``+0`` or
    ``-0``).
    """
    plain = _format_decimal(value)
    if plain == "0":
        return "0"
    if plain.startswith("-"):
        return plain
    return "+" + plain


def _display_label(label: str) -> str:
    """Escape control characters in one persisted label for display.

    Backslash becomes ``\\\\``, CR/LF/TAB become ``\\r``/``\\n``/``\\t``,
    and every other C0 control character (U+0000..U+001F) plus DEL
    (U+007F) becomes ``\\uXXXX`` with uppercase hexadecimal digits. All
    other characters, including Cyrillic, Latin, spaces, and
    punctuation, pass through unchanged, so a label always occupies
    exactly one logical line of the report.
    """
    pieces: list[str] = []
    for character in label:
        if character == "\\":
            pieces.append("\\\\")
        elif character == "\r":
            pieces.append("\\r")
        elif character == "\n":
            pieces.append("\\n")
        elif character == "\t":
            pieces.append("\\t")
        else:
            code = ord(character)
            if code <= 0x1F or code == 0x7F:
                pieces.append(f"\\u{code:04X}")
            else:
                pieces.append(character)
    return "".join(pieces)


def render_monthly_report(report: MonthlyFinanceReport) -> str:
    """Render one monthly finance report as deterministic Russian plain text.

    ``report`` must be a real :class:`MonthlyFinanceReport` (anything
    else is rejected with :class:`TypeError`); it is never mutated and
    none of its values are recalculated, re-summed, or reordered.
    Categories render in the report's own order and sources in each
    category's own order.

    A non-empty report (``transaction_count != 0`` or categories
    present) renders the uppercase identity header
    (``💰 FINANCE | <МЕСЯЦ> <ГОД>``), the four emoji-led month totals,
    then one compact block per category: the ``📂`` category heading
    with its four totals, one ``🔹 Источники`` marker, and one bullet
    name line plus one compact detail line per source. A fully empty
    month (``transaction_count == 0`` and ``categories == ()``) renders
    the header, zero totals, and ``📭 Операций за месяц нет.`` instead
    of the category blocks.

    The result uses spaces only (never tabs), contains no trailing
    whitespace, no markup of any kind, and no final newline.

    Raises
    ------
    TypeError
        If ``report`` is not a :class:`MonthlyFinanceReport` or any
        money field holds a non-finite Decimal.
    """
    if not isinstance(report, MonthlyFinanceReport):
        raise TypeError(
            f"report must be a MonthlyFinanceReport, got {type(report).__name__!r}"
        )

    month_name = _MONTH_NAMES[report.month]
    lines: list[str] = [
        f"💰 FINANCE | {month_name} {report.year}",
        "",
        f"📈 Доход: {_format_decimal(report.income_usdt)} USDT",
        f"📉 Расход: {_format_decimal(report.expense_usdt)} USDT",
        f"⚖️ Итог: {_format_net(report.net_usdt)} USDT",
        f"🧾 Операций: {report.transaction_count}",
    ]

    if report.transaction_count == 0 and report.categories == ():
        lines.append("")
        lines.append("📭 Операций за месяц нет.")
        return "\n".join(lines)

    for category in report.categories:
        lines.append("")
        lines.append(f"📂 {_display_label(category.category)}")
        lines.append(f"Доход: {_format_decimal(category.income_usdt)} USDT")
        lines.append(f"Расход: {_format_decimal(category.expense_usdt)} USDT")
        lines.append(f"Итог: {_format_net(category.net_usdt)} USDT")
        lines.append(f"Операций: {category.transaction_count}")
        lines.append("")
        lines.append("🔹 Источники")
        for source in category.sources:
            lines.append(f"• {_display_label(source.source)}")
            lines.append(
                f"  доход {_format_decimal(source.income_usdt)}"
                f" · расход {_format_decimal(source.expense_usdt)}"
                f" · итог {_format_net(source.net_usdt)} USDT"
                f" · операций {source.transaction_count}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# stage G1: deterministic transaction-list rendering
# ---------------------------------------------------------------------------

#: Title of the "recent transactions" view.
_RECENT_TITLE: Final[str] = "🧾 FINANCE | ПОСЛЕДНИЕ ОПЕРАЦИИ"

#: Title prefix of every exact-view (date/month) transaction list.
_VIEW_TITLE_PREFIX: Final[str] = "🧾 FINANCE | ОПЕРАЦИИ"

#: The single deterministic empty-state marker of every G1 view.
_EMPTY_LIST_MARKER: Final[str] = "📭 Операций нет."

#: Direction icon policy: income is green, expense is red.
_DIRECTION_ICONS: Final[dict[Direction, str]] = {
    Direction.INCOME: "🟢",
    Direction.EXPENSE: "🔴",
}

#: Direction sign policy for the money fragment of a card.
_DIRECTION_SIGNS: Final[dict[Direction, str]] = {
    Direction.INCOME: "+",
    Direction.EXPENSE: "-",
}


def _format_day_month_year(value: date) -> str:
    """Render one calendar date as the deterministic ``DD.MM.YYYY`` text.

    Built directly from the date attributes with fixed-width integer
    formatting only: no locale machinery, no ``strftime``, and no clock.
    """
    return f"{value.day:02d}.{value.month:02d}.{value.year:04d}"


def _render_transaction_card(transaction: Transaction) -> str:
    """Render exactly one transaction as a compact multi-line card.

    The card is::

        <icon> #<transaction_id> | <sign><amount> USDT | DD.MM.YYYY
        <category> • <source>
        💬 <comment>            (only when a comment exists)

    The persisted ``transaction_id`` string is displayed exactly as
    stored; a transaction without a real persisted id fails
    deterministically instead of rendering a fake identifier.
    """
    transaction_id = transaction.transaction_id
    if transaction_id is None:
        raise ValueError(
            "a rendered transaction must carry its persisted transaction_id;"
            " an unpersisted transaction cannot be displayed"
        )
    icon = _DIRECTION_ICONS[transaction.direction]
    sign = _DIRECTION_SIGNS[transaction.direction]
    amount = f"{sign}{_format_decimal(transaction.amount_usdt)} USDT"
    day = _format_day_month_year(transaction.transaction_date)
    lines = [
        f"{icon} #{transaction_id} | {amount} | {day}",
        f"{_display_label(transaction.category)} • {_display_label(transaction.source)}",
    ]
    if transaction.comment is not None:
        lines.append(f"💬 {_display_label(transaction.comment)}")
    return "\n".join(lines)


def _render_transaction_view(
    title: str,
    transactions: Sequence[Transaction],
) -> str:
    """Render one titled transaction list: title, blank line, cards.

    Cards render in the exact supplied sequence with exactly one blank
    line between them; an empty sequence renders the single
    deterministic empty-state marker instead of cards. A non-iterable
    ``transactions`` value is rejected with :class:`TypeError` before
    the empty-state check.
    """
    items = tuple(transactions)
    if not items:
        return f"{title}\n\n{_EMPTY_LIST_MARKER}"

    cards: list[str] = []
    for transaction in items:
        if not isinstance(transaction, Transaction):
            raise TypeError(
                "transactions must contain Transaction objects,"
                f" got {type(transaction).__name__!r}"
            )
        cards.append(_render_transaction_card(transaction))
    return "\n".join([title, "", "\n\n".join(cards)])


def render_recent_transactions(transactions: Sequence[Transaction]) -> str:
    """Render the "last N transactions" view as deterministic plain text.

    ``transactions`` is the exact tuple returned by the accepted D1
    operation :func:`~hermes_finance.operations.list_recent_transactions`;
    its newest-first order is displayed verbatim and never re-sorted.
    Every element must be a real
    :class:`~hermes_finance.domain.Transaction` with a persisted
    ``transaction_id``; lookalikes are rejected with :class:`TypeError`
    and unpersisted transactions with :class:`ValueError`.

    An empty tuple renders the title and ``📭 Операций нет.`` The
    result contains no trailing whitespace and no final newline.
    """
    return _render_transaction_view(_RECENT_TITLE, transactions)


def render_transactions_by_date(
    transactions: Sequence[Transaction],
    transaction_date: date,
) -> str:
    """Render the "transactions of one exact date" view.

    ``transaction_date`` must be a plain ``datetime.date`` supplied
    explicitly by the caller (``datetime`` instances and arbitrary
    non-date values are rejected through the accepted domain
    validation); it is used only for the ``DD.MM.YYYY`` title and is
    never derived from any clock.

    ``transactions`` is the exact tuple returned by the accepted D1
    operation :func:`~hermes_finance.operations.list_transactions_by_date`;
    its order is displayed verbatim. Input, empty-state, and output
    contracts match :func:`render_recent_transactions`.
    """
    validated_date = require_calendar_date(transaction_date, "transaction_date")
    title = f"{_VIEW_TITLE_PREFIX} {_format_day_month_year(validated_date)}"
    return _render_transaction_view(title, transactions)


def render_transactions_by_month(
    transactions: Sequence[Transaction],
    *,
    year: int,
    month: int,
) -> str:
    """Render the "transactions of one exact calendar month" view.

    ``month`` must be a real ``int`` between 1 and 12 (``bool`` is
    rejected) and ``year`` a real ``int``; both are used only for the
    uppercase Russian month title and are never derived from any
    clock. ``transactions`` is the exact tuple returned by the accepted
    D1 operation
    :func:`~hermes_finance.operations.list_transactions_by_month`;
    its order is displayed verbatim. Input, empty-state, and output
    contracts match :func:`render_recent_transactions`.
    """
    if isinstance(month, bool) or not isinstance(month, int):
        raise TypeError(f"month must be an int, got {type(month).__name__!r}")
    month_name = _MONTH_NAMES.get(month)
    if month_name is None:
        raise ValueError(f"month must be between 1 and 12, got {month!r}")
    if isinstance(year, bool) or not isinstance(year, int):
        raise TypeError(f"year must be an int, got {type(year).__name__!r}")
    title = f"{_VIEW_TITLE_PREFIX} {month_name} {year}"
    return _render_transaction_view(title, transactions)


# ---------------------------------------------------------------------------
# stage G2: deterministic filtered monthly summary rendering
# ---------------------------------------------------------------------------

#: The single deterministic "no data" marker of every G2 filtered view.
#: It reports that the requested label does not exist in the selected
#: month, deliberately distinguishable from an existing label whose
#: authoritative E1 totals happen to be exact zeros.
_FILTERED_NO_DATA_MARKER: Final[str] = "📭 Данных нет."


def _require_summary_year_month(year: int, month: int) -> str:
    """Validate the summary ``year``/``month`` and return the month name.

    Mirrors the accepted G1 month-title validation exactly: a real
    ``int`` month between 1 and 12 (``bool`` rejected) and a real
    ``int`` year, with no clock, no locale, and no coercion.
    """
    if isinstance(month, bool) or not isinstance(month, int):
        raise TypeError(f"month must be an int, got {type(month).__name__!r}")
    month_name = _MONTH_NAMES.get(month)
    if month_name is None:
        raise ValueError(f"month must be between 1 and 12, got {month!r}")
    if isinstance(year, bool) or not isinstance(year, int):
        raise TypeError(f"year must be an int, got {type(year).__name__!r}")
    return month_name


def _summary_lines(
    year: int,
    month: int,
    search_line: str,
    values: tuple[str, ...] | None,
) -> str:
    """Assemble one G2 summary card: header, search line, four values.

    ``values`` holds the four already-rendered metric lines taken
    verbatim from the supplied authoritative report object, or ``None``
    for the explicit "no data" state of a missing label. The result has
    no trailing whitespace and no final newline.
    """
    month_name = _require_summary_year_month(year, month)
    lines: list[str] = [
        f"💰 FINANCE | {month_name} {year}",
        "",
        search_line,
        "",
    ]
    if values is None:
        lines.append(_FILTERED_NO_DATA_MARKER)
    else:
        lines.extend(values)
    return "\n".join(lines)


def render_monthly_category_summary(
    *,
    year: int,
    month: int,
    category: str,
    category_report: CategoryReport | None,
) -> str:
    """Render one filtered category summary as deterministic plain text.

    ``category`` is normalised with the accepted required-text
    surrounding-whitespace normalisation (a non-string raises
    :class:`TypeError`, a blank value raises :class:`ValueError`) and is
    displayed through the accepted visible-escape policy.
    ``category_report`` must be the authoritative immutable
    :class:`CategoryReport` selected from the month's E1 report by the
    accepted G2 selection layer, or ``None`` when the label does not
    exist in that month; lookalike objects are rejected with
    :class:`TypeError`.

    A found report renders the ``🔎 Категория:`` search line and the
    four emoji-led metrics (income, expense, net, operation count)
    taken verbatim from the report object: nothing is recalculated,
    re-summed, or reformatted beyond the accepted exact money
    formatting. ``None`` renders the same header and search line
    followed by ``📭 Данных нет.`` -- an explicit missing-label state,
    never a manufactured zero report.

    The result uses spaces only, contains no trailing whitespace, no
    markup of any kind, and no final newline.
    """
    label = normalize_required_text(category, "category")
    if category_report is not None and not isinstance(
        category_report, CategoryReport
    ):
        raise TypeError(
            "category_report must be a CategoryReport or None,"
            f" got {type(category_report).__name__!r}"
        )
    search_line = f"🔎 Категория: {_display_label(label)}"
    values = (
        None
        if category_report is None
        else (
            f"📈 Доход: {_format_decimal(category_report.income_usdt)} USDT",
            f"📉 Расход: {_format_decimal(category_report.expense_usdt)} USDT",
            f"⚖️ Итог: {_format_net(category_report.net_usdt)} USDT",
            f"🧾 Операций: {category_report.transaction_count}",
        )
    )
    return _summary_lines(year, month, search_line, values)


def render_monthly_source_summary(
    *,
    year: int,
    month: int,
    category: str,
    source: str,
    source_report: SourceReport | None,
) -> str:
    """Render one filtered category-local source summary as plain text.

    ``category`` and ``source`` are normalised with the accepted
    required-text surrounding-whitespace normalisation (non-strings
    raise :class:`TypeError`, blank values raise :class:`ValueError`)
    and are displayed through the accepted visible-escape policy as the
    ``🔎 <category> • <source>`` search line. ``source_report`` must be
    the authoritative immutable category-local :class:`SourceReport`
    selected inside one category of the month's E1 report by the
    accepted G2 selection layer, or ``None`` when the category or the
    source does not exist in that month; lookalike objects are rejected
    with :class:`TypeError`.

    A found report renders the four emoji-led metrics (income, expense,
    net, operation count) taken verbatim from the report object:
    nothing is recalculated, re-summed, or reformatted beyond the
    accepted exact money formatting, and sources of other categories
    never contribute. ``None`` renders the same header and search line
    followed by ``📭 Данных нет.`` -- an explicit missing-label state,
    never a manufactured zero report.

    The result uses spaces only, contains no trailing whitespace, no
    markup of any kind, and no final newline.
    """
    category_label = normalize_required_text(category, "category")
    source_label = normalize_required_text(source, "source")
    if source_report is not None and not isinstance(source_report, SourceReport):
        raise TypeError(
            "source_report must be a SourceReport or None,"
            f" got {type(source_report).__name__!r}"
        )
    search_line = (
        f"🔎 {_display_label(category_label)} • {_display_label(source_label)}"
    )
    values = (
        None
        if source_report is None
        else (
            f"📈 Доход: {_format_decimal(source_report.income_usdt)} USDT",
            f"📉 Расход: {_format_decimal(source_report.expense_usdt)} USDT",
            f"⚖️ Итог: {_format_net(source_report.net_usdt)} USDT",
            f"🧾 Операций: {source_report.transaction_count}",
        )
    )
    return _summary_lines(year, month, search_line, values)
