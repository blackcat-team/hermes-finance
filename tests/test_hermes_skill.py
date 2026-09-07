"""Bounded contract tests for the standalone Hermes skill source.

All tests are deterministic: no network, no subprocesses, no wall-clock
time, no randomness, no filesystem mutation. The skill is validated as
static version-controlled content only.

The frontmatter is validated structurally against the canonical Hermes
skill schema (v0.21.0): ``metadata.hermes.requires_tools``,
``metadata.hermes.config`` entries, and top-level
``required_environment_variables``. Because the project deliberately has
no YAML dependency, this file contains a minimal bounded parser for the
exact YAML subset used by the skill frontmatter (block mappings, block
sequences, and scalars); it is not a general YAML library. Scalar values
are additionally validated against the YAML plain-scalar restriction that
Hermes/PyYAML enforces: an unquoted scalar must not contain an embedded
": " (mapping indicator), because a PyYAML parse failure makes Hermes
fall back to a flat parser that loses ``metadata.hermes.config``.

Since F4R2 the skill must describe the live production architecture: fast
Telegram transaction persistence is owned by the separate native
``blackcat-finance-fastpath`` plugin (stage F5, live in production), while
this skill owns only the monthly report. The tests pin that the skill
acknowledges the native fast path, never claims to run
``hermes-finance ingest-telegram`` itself, never fabricates Telegram
provenance, never reports false success, and no longer carries any stale
pre-F5 wording that would make the model tell the user that automatic
saving is unavailable.

Since G3 the skill maps the four planned relative periods («сегодня»,
«вчера», «этот месяц», «прошлый месяц») to the canonical relative CLI
commands carrying the configured ``blackcat_finance.business_timezone``,
while the model itself never derives any calendar date and the skill
declares no default timezone.

Since H1 the skill additionally owns the transaction mutation commands:
amount-only correction and soft deletion through
``hermes-finance transaction edit-amount`` / ``transaction delete`` with
the exact-ID and ``--last`` targets. The tests pin the direct
natural-language mappings, the ID-safety rules (IDs only from Finance
output or user input, never Telegram provenance), the amount-only
semantics with a preserved direction, the refusal to emulate direction
changes, and the no-false-success rule: a mutation is successful only
after the CLI returns its deterministic JSON disposition.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Final

import pytest

REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

#: Version-controlled source location of the standalone skill.
SKILL_PATH: Final[Path] = (
    REPOSITORY_ROOT / "integrations" / "hermes" / "skills" / "blackcat-finance"
    / "SKILL.md"
)

#: Production deployment target (user-owned Hermes extension storage).
PRODUCTION_TARGET: Final[str] = "~/.hermes/skills/blackcat-finance/"

#: Infrastructure markers that must never appear in skill architecture.
FORBIDDEN_INFRASTRUCTURE: Final[tuple[str, ...]] = (
    "http://",
    "https://",
    "systemd",
    "cron",
    "docker",
    "kubernetes",
    "uvicorn",
    "gunicorn",
    "fastapi",
    "flask",
    "nginx",
    "mcp",
)

#: SQL statement markers that must never appear as instructions.
FORBIDDEN_SQL_MARKERS: Final[tuple[str, ...]] = (
    "SELECT ",
    "INSERT INTO",
    "UPDATE ",
    "DELETE FROM",
)

#: Fields supported by the canonical ``metadata.hermes.config`` entry schema.
SUPPORTED_CONFIG_FIELDS: Final[frozenset[str]] = frozenset(
    {"key", "description", "default", "prompt"}
)

#: Fields supported by the canonical ``required_environment_variables`` schema.
SUPPORTED_ENV_FIELDS: Final[frozenset[str]] = frozenset(
    {"name", "prompt", "required_for"}
)

SKILL_TEXT: Final[str] = SKILL_PATH.read_text(encoding="utf-8")


def split_frontmatter(text: str) -> tuple[str, str]:
    """Split a skill document into (frontmatter text, body text).

    Raises AssertionError for a document without a well-formed
    ``---``-delimited frontmatter block.
    """
    lines = text.splitlines()
    assert lines, "skill document is empty"
    assert lines[0] == "---", "skill document must start with a --- delimiter"
    try:
        closing = lines[1:].index("---") + 1
    except ValueError as error:
        raise AssertionError("frontmatter is not closed") from error
    frontmatter = lines[1:closing]
    assert frontmatter, "frontmatter block is empty"
    assert "---" not in lines[closing + 1 :], "frontmatter is not closed exactly once"
    return "\n".join(frontmatter), "\n".join(lines[closing + 1 :])


def paragraphs(body: str) -> list[str]:
    """Split a markdown body into blank-line-separated paragraph blocks."""
    return [block.strip() for block in re.split(r"\n\s*\n", body) if block.strip()]


def _is_indented_diagram(paragraph: str) -> bool:
    """Return True for an indented architecture diagram block.

    ``paragraphs`` strips the leading indent of a block's first line, so a
    diagram is recognized by its arrow chain plus indented continuation
    lines. Prose paragraphs in this skill are hard-wrapped without any
    indented continuation, so they never match.
    """
    lines = paragraph.splitlines()
    if "->" not in paragraph or len(lines) < 2:
        return False
    return all(line.startswith(" ") for line in lines[1:])


def _indent_of(line: str) -> int:
    """Return the leading-space indent width of a frontmatter line."""
    stripped = line.lstrip(" ")
    assert stripped, "frontmatter must not contain blank lines"
    assert "\t" not in line, "frontmatter must not use tab indentation"
    return len(line) - len(stripped)


def _scalar(value: str) -> str:
    """Validate and unwrap one frontmatter scalar (bounded YAML rule).

    Mirrors the PyYAML behavior Hermes relies on for the tiny subset this
    skill uses: a double- or single-quoted scalar is unwrapped verbatim,
    while an unquoted plain scalar must not contain an embedded ``": "``
    sequence. In real YAML, ``": "`` inside a plain scalar is a mapping
    indicator and makes the document invalid; when Hermes/PyYAML rejects
    the frontmatter, its fallback parser loses the nested
    ``metadata.hermes.config`` structure entirely (the F4R1 runtime
    defect). This is deliberately not a general YAML implementation.
    """
    assert value, "empty frontmatter scalar"
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    assert ": " not in value, (
        "unquoted plain scalar contains YAML-invalid ': ' "
        "(quote the scalar): " f"{value!r}"
    )
    return value


def _parse_mapping(
    entries: list[tuple[int, str]], index: int, indent: int
) -> tuple[dict[str, Any], int]:
    """Parse a block mapping starting at ``entries[index]`` at ``indent``.

    Each entry must be ``key: value`` or ``key:`` followed by a deeper
    nested block. Returns the mapping and the index after the block.
    """
    result: dict[str, Any] = {}
    while index < len(entries) and entries[index][0] == indent:
        text = entries[index][1]
        key, separator, rest = text.partition(":")
        assert separator, f"invalid mapping line: {text!r}"
        key = key.strip()
        rest = rest.strip()
        assert key not in result, f"duplicate frontmatter key: {key}"
        index += 1
        if rest:
            result[key] = _scalar(rest)
        elif index < len(entries) and entries[index][0] > indent:
            nested, index = _parse_value(entries, index, entries[index][0])
            result[key] = nested
        else:
            result[key] = None
    return result, index


def _parse_value(
    entries: list[tuple[int, str]], index: int, indent: int
) -> tuple[Any, int]:
    """Parse one block (mapping or sequence) starting at ``entries[index]``."""
    if entries[index][1].startswith("- "):
        items: list[Any] = []
        while (
            index < len(entries)
            and entries[index][0] == indent
            and entries[index][1].startswith("- ")
        ):
            item_text = entries[index][1][2:].strip()
            index += 1
            if ":" in item_text:
                # Mapping sequence item: "- key: value" plus following
                # deeper lines form one mapping.
                virtual: list[tuple[int, str]] = [(indent + 2, item_text)]
                while index < len(entries) and entries[index][0] > indent:
                    virtual.append(entries[index])
                    index += 1
                item, _ = _parse_mapping(virtual, 0, indent + 2)
                items.append(item)
            else:
                items.append(item_text)
        assert items, "empty sequence block"
        return items, index
    return _parse_mapping(entries, index, indent)


def parse_frontmatter(text: str) -> dict[str, Any]:
    """Structurally parse the bounded YAML subset of the skill frontmatter."""
    entries: list[tuple[int, str]] = [
        (_indent_of(line), line.strip()) for line in text.splitlines()
    ]
    parsed, next_index = _parse_mapping(entries, 0, 0)
    assert next_index == len(entries), "unconsumed frontmatter lines"
    return parsed


FRONTMATTER_TEXT: Final[str] = split_frontmatter(SKILL_TEXT)[0]
FRONTMATTER: Final[dict[str, Any]] = parse_frontmatter(FRONTMATTER_TEXT)
BODY: Final[str] = split_frontmatter(SKILL_TEXT)[1]
BODY_PARAGRAPHS: Final[list[str]] = paragraphs(BODY)


def _config_entries() -> list[dict[str, Any]]:
    """Return the ``metadata.hermes.config`` entries as mappings."""
    config = FRONTMATTER["metadata"]["hermes"]["config"]
    assert isinstance(config, list)
    for entry in config:
        assert isinstance(entry, dict)
    return config


def _cli_path_config_entry() -> dict[str, Any]:
    """Return the ``blackcat_finance.cli_path`` config entry."""
    entries = [entry for entry in _config_entries() if entry["key"] == "blackcat_finance.cli_path"]
    assert len(entries) == 1, "exactly one blackcat_finance.cli_path entry required"
    return entries[0]


def _business_timezone_config_entry() -> dict[str, Any]:
    """Return the ``blackcat_finance.business_timezone`` config entry."""
    entries = [
        entry
        for entry in _config_entries()
        if entry["key"] == "blackcat_finance.business_timezone"
    ]
    assert (
        len(entries) == 1
    ), "exactly one blackcat_finance.business_timezone entry required"
    return entries[0]


def _env_entries() -> list[dict[str, Any]]:
    """Return the top-level ``required_environment_variables`` entries."""
    variables = FRONTMATTER["required_environment_variables"]
    assert isinstance(variables, list)
    for entry in variables:
        assert isinstance(entry, dict)
    return variables


class TestSkillSourceLocation:
    def test_skill_file_exists_at_integration_source_path(self) -> None:
        assert SKILL_PATH.is_file()

    def test_production_target_is_user_owned_extension_storage(self) -> None:
        assert PRODUCTION_TARGET in SKILL_TEXT


class TestFrontmatterIdentity:
    def test_frontmatter_is_structurally_parseable(self) -> None:
        assert FRONTMATTER

    def test_skill_name_is_blackcat_finance(self) -> None:
        assert FRONTMATTER["name"] == "blackcat-finance"

    def test_description_present(self) -> None:
        description = FRONTMATTER["description"]
        assert isinstance(description, str) and description.strip()


class TestFrontmatterYamlSafety:
    """Regression guard for the F4R1 runtime defect.

    Hermes v0.21.0 parses the frontmatter with PyYAML. An unquoted plain
    scalar containing an embedded ``": "`` is invalid YAML, so PyYAML
    fails, the Hermes fallback parser kicks in, and the nested
    ``metadata.hermes.config`` declaration is lost (declared config vars
    become ``[]``). These tests pin the committed frontmatter to the
    YAML-safe subset without introducing a YAML dependency.
    """

    def test_description_scalar_is_quoted_or_plain_yaml_safe(self) -> None:
        line = next(
            line
            for line in FRONTMATTER_TEXT.splitlines()
            if line.startswith("description:")
        )
        value = line[len("description:") :].strip()
        assert value, "description must have a value"
        if not (value[0] == value[-1] and value[0] in "\"'"):
            assert ": " not in value, (
                "unquoted description contains YAML-invalid ': ': " f"{value!r}"
            )

    def test_description_meaning_preserved(self) -> None:
        assert FRONTMATTER["description"] == (
            "Personal bookkeeping through the external hermes-finance CLI. "
            "Monthly finance report, filtered monthly summaries, "
            "transaction lists, and transaction corrections/deletion over "
            "the Finance ledger."
        )

    def test_unquoted_plain_scalar_with_colon_space_rejected(self) -> None:
        with pytest.raises(AssertionError, match="YAML-invalid"):
            parse_frontmatter("description: a b: c")

    def test_exact_pre_f4r1_unquoted_description_rejected(self) -> None:
        # The exact frontmatter accepted before the F4R1 fix: PyYAML
        # rejects it, Hermes falls back to its flat parser, and
        # metadata.hermes.config is lost. It must fail the scalar rule.
        pre_f4r1_frontmatter = (
            "name: blackcat-finance\n"
            "description: Personal bookkeeping through the external "
            "hermes-finance CLI. Stage F4 capability: monthly finance "
            "report.\n"
        )
        with pytest.raises(AssertionError, match="YAML-invalid"):
            parse_frontmatter(pre_f4r1_frontmatter)

    def test_quoted_scalar_with_colon_space_accepted(self) -> None:
        parsed = parse_frontmatter('description: "a b: c"')
        assert parsed["description"] == "a b: c"

    def test_committed_frontmatter_parses_under_scalar_rule(self) -> None:
        # Re-parses the committed source through the strengthened rule; the
        # pre-F4R1 source (unquoted ': ' in description) fails right here.
        reparsed = parse_frontmatter(FRONTMATTER_TEXT)
        assert reparsed == FRONTMATTER


class TestFrontmatterToolRequirement:
    def test_requires_tools_declared_under_metadata_hermes(self) -> None:
        requires_tools = FRONTMATTER["metadata"]["hermes"]["requires_tools"]
        assert requires_tools == ["terminal"]

    def test_no_custom_requires_shape(self) -> None:
        assert "requires" not in FRONTMATTER
        hermes = FRONTMATTER["metadata"]["hermes"]
        assert "tools" not in hermes
        assert "env" not in hermes


class TestFrontmatterCliPathConfig:
    def test_config_declared_under_metadata_hermes(self) -> None:
        entry = _cli_path_config_entry()
        assert entry["key"] == "blackcat_finance.cli_path"

    def test_config_entry_has_key_and_description(self) -> None:
        entry = _cli_path_config_entry()
        description = entry["description"]
        assert isinstance(description, str) and description.strip()

    def test_config_entry_uses_only_supported_fields(self) -> None:
        entry = _cli_path_config_entry()
        assert set(entry) <= SUPPORTED_CONFIG_FIELDS

    def test_no_hard_coded_production_default_path(self) -> None:
        entry = _cli_path_config_entry()
        assert "default" not in entry

    def test_no_concrete_database_or_executable_path_in_frontmatter(self) -> None:
        assert not re.search(r"\.db\b|\.sqlite\b|\.exe\b", FRONTMATTER_TEXT)


class TestFrontmatterBusinessTimezoneConfig:
    """G3: ``blackcat_finance.business_timezone`` is declared, described,
    and has no default.

    Relative Finance periods are resolved by the CLI in this configured
    IANA business timezone. It is a separate non-secret config value
    from ``blackcat_finance.cli_path``, and no production timezone is
    hard-coded anywhere in the skill.
    """

    def test_config_declared_under_metadata_hermes(self) -> None:
        entry = _business_timezone_config_entry()
        assert entry["key"] == "blackcat_finance.business_timezone"

    def test_config_entry_has_description_and_prompt(self) -> None:
        entry = _business_timezone_config_entry()
        assert isinstance(entry["description"], str) and entry["description"].strip()
        assert isinstance(entry["prompt"], str) and entry["prompt"].strip()

    def test_config_entry_uses_only_supported_fields(self) -> None:
        assert set(_business_timezone_config_entry()) <= SUPPORTED_CONFIG_FIELDS

    def test_no_default_timezone_value(self) -> None:
        assert "default" not in _business_timezone_config_entry()

    def test_description_names_iana_and_relative_periods(self) -> None:
        description = _business_timezone_config_entry()["description"]
        assert "IANA" in description
        assert "relative periods" in description

    def test_no_hard_coded_production_timezone_in_skill(self) -> None:
        assert "Europe/Moscow" not in SKILL_TEXT


class TestFrontmatterEnvironmentVariables:
    def test_required_environment_variables_is_top_level(self) -> None:
        entries = _env_entries()
        assert entries

    def test_db_path_environment_variable_declared(self) -> None:
        matching = [entry for entry in _env_entries() if entry["name"] == "HERMES_FINANCE_DB_PATH"]
        assert len(matching) == 1

    def test_db_path_entry_has_prompt_and_required_for(self) -> None:
        entry = next(
            entry for entry in _env_entries() if entry["name"] == "HERMES_FINANCE_DB_PATH"
        )
        assert isinstance(entry["prompt"], str) and entry["prompt"].strip()
        assert entry["required_for"] == "Hermes Finance CLI database access"

    def test_env_entries_use_only_supported_fields(self) -> None:
        for entry in _env_entries():
            assert set(entry) <= SUPPORTED_ENV_FIELDS

    def test_no_actual_db_path_value_in_frontmatter(self) -> None:
        for path_marker in (
            re.compile(r"[A-Za-z]:[\\/]"),
            re.compile(r"/home/"),
            re.compile(r"/root/"),
            re.compile(r"~/"),
        ):
            assert path_marker.search(FRONTMATTER_TEXT) is None


class TestMonthlyReportProcedure:
    def test_month_command_documented(self) -> None:
        assert "month --year YYYY --month M" in BODY
        assert "month --year 2026 --month 8" in BODY

    def test_terminal_execution_documented(self) -> None:
        assert "terminal" in BODY.lower()

    def test_cli_path_placeholder_used_not_hardcoded_checkout(self) -> None:
        assert "<cli-path>" in BODY
        assert "blackcat_finance.cli_path" in BODY

    def test_report_output_is_authoritative(self) -> None:
        assert "authoritative" in BODY
        assert "do not recalculate totals" in BODY
        assert "do not change amounts" in BODY
        assert "do not re-group categories or sources" in BODY
        assert "do not invent missing transactions" in BODY

    def test_natural_language_intent_mapping_documented(self) -> None:
        assert "Покажи отчёт за август 2026" in BODY
        assert "Сколько было за август 2026?" in BODY


class TestTransactionListProcedures:
    """G1: the skill teaches the transaction-list commands exactly."""

    def test_recent_command_documented(self) -> None:
        flattened = " ".join(BODY.split())
        assert "`<cli-path> transactions recent`" in flattened
        assert "`<cli-path> transactions recent --limit N`" in flattened
        assert "`<cli-path> transactions recent --limit 5`" in flattened

    def test_recent_default_limit_documented(self) -> None:
        flattened = " ".join(BODY.split())
        assert "last 10 by default" in flattened

    def test_date_command_documented(self) -> None:
        flattened = " ".join(BODY.split())
        assert "`<cli-path> transactions date --date YYYY-MM-DD`" in flattened
        assert "`<cli-path> transactions date --date 2026-09-05`" in flattened

    def test_month_list_command_documented(self) -> None:
        flattened = " ".join(BODY.split())
        assert "`<cli-path> transactions month --year YYYY --month M`" in flattened
        assert "`<cli-path> transactions month --year 2026 --month 9`" in flattened

    def test_list_output_is_authoritative(self) -> None:
        flattened = " ".join(BODY.split())
        assert "the list itself is authoritative and MUST NOT be altered" in flattened
        assert "do not re-sort operations" in flattened
        assert "do not change or hide transaction identifiers (`#<id>` lines)" in (
            flattened
        )

    def test_recent_intent_documented(self) -> None:
        assert "«Покажи последние операции» -> `<cli-path> transactions recent`" in BODY

    def test_recent_n_intent_documented(self) -> None:
        assert (
            "«Покажи последние 5 операций» ->"
            " `<cli-path> transactions recent --limit 5`" in BODY
        )

    def test_exact_date_intent_documented(self) -> None:
        assert (
            "«Покажи операции за 5 сентября 2026» ->"
            " `<cli-path> transactions date --date 2026-09-05`" in BODY
        )
        assert (
            "«Что записано 5 сентября 2026» ->"
            " `<cli-path> transactions date --date 2026-09-05`" in BODY
        )

    def test_exact_month_list_intent_documented(self) -> None:
        assert (
            "«Покажи операции за сентябрь 2026» ->"
            " `<cli-path> transactions month --year 2026 --month 9`" in BODY
        )

    def test_monthly_report_intent_remains_distinct(self) -> None:
        """LIST OF OPERATIONS and MONTHLY AGGREGATE REPORT stay distinct."""
        assert (
            "«Покажи отчёт за сентябрь 2026» -> `<cli-path> month --year 2026 --month 9`"
            in BODY
        )
        # The distinction is stated explicitly.
        flattened = " ".join(BODY.split())
        assert "A transaction list and the monthly report are distinct capabilities" in (
            flattened
        )
        assert "aggregated monthly report with totals" in flattened

    def test_transaction_list_chain_documented(self) -> None:
        flattened = " ".join(BODY.split())
        for step in (
            "Telegram Finance topic",
            "blackcat-finance skill",
            "terminal",
            "hermes-finance transactions",
            "Finance core",
            "SQLite",
        ):
            assert step in flattened, step

    def test_read_choose_mutate_capabilities_owned(self) -> None:
        """H1: four natural-language read/mutation capabilities are owned."""
        flattened = " ".join(BODY.split())
        assert "owns exactly four natural-language read/mutation capabilities" in (
            flattened
        )
        assert "monthly finance report (aggregated totals per month)" in flattened
        assert (
            "filtered monthly summaries (one exact category, or one exact"
            " category and its source, of one exact month)" in flattened
        )
        assert "transaction lists (individual stored operations)" in flattened
        assert (
            "transaction corrections/deletion (amount-only correction and"
            " soft deletion of already stored operations)" in flattened
        )


class TestTransactionCorrectionsAndDeletion:
    """H1: the skill teaches the mutation commands exactly.

    «Удали последнюю операцию», «Исправь последнюю сумму на 12»,
    «Удали #15», and «Исправь сумму #15 на 12» map to the canonical
    mutation commands. The model never invents an ID, never uses
    Telegram provenance as a transaction ID, never decides the direction
    of the corrected amount, never emulates an unsupported direction
    change, and never reports success without the CLI's deterministic
    JSON disposition.
    """

    def test_delete_last_mapping_documented(self) -> None:
        assert (
            "«Удали последнюю операцию» ->"
            " `<cli-path> transaction delete --last`" in BODY
        )

    def test_edit_last_mapping_documented(self) -> None:
        assert (
            "«Исправь последнюю сумму на 12» ->"
            " `<cli-path> transaction edit-amount --last --amount 12`" in BODY
        )

    def test_delete_id_mapping_documented(self) -> None:
        assert "«Удали #15» -> `<cli-path> transaction delete --id 15`" in BODY

    def test_edit_id_mapping_documented(self) -> None:
        assert (
            "«Исправь сумму #15 на 12» ->"
            " `<cli-path> transaction edit-amount --id 15 --amount 12`" in BODY
        )

    def test_last_resolved_by_finance_not_by_the_model(self) -> None:
        flattened = " ".join(BODY.split())
        assert (
            "`--last` is resolved by the finance CLI itself with exactly the"
            " same selection as `transactions recent --limit 1`" in flattened
        )
        assert (
            "NEVER determine yourself which transaction is"
            ' "last": pass `--last` and let Finance resolve the target'
            in flattened
        )

    def test_ids_only_from_finance_output_or_user_input(self) -> None:
        flattened = " ".join(BODY.split())
        assert (
            "Transaction IDs come only from Finance CLI output (the `#<id>`"
            " lines of a transaction list) or from explicit user input"
            in flattened
        )
        assert "do not invent an ID" in flattened
        assert "do not guess an ID" in flattened

    def test_telegram_provenance_never_used_as_transaction_id(self) -> None:
        flattened = " ".join(BODY.split())
        assert (
            "do not use a Telegram `message_id`, `update_id`, or any other"
            " Telegram provenance value as a transaction ID" in flattened
        )

    def test_unclear_target_shows_a_list_first(self) -> None:
        flattened = " ".join(BODY.split())
        assert (
            "if the user refers to an operation without an ID and without"
            ' "last" wording, show a transaction list first and let the user'
            " choose" in flattened
        )

    def test_amount_is_positive_magnitude_only(self) -> None:
        flattened = " ".join(BODY.split())
        assert (
            "`--amount` is always the positive magnitude. The stored"
            " income/expense direction is preserved by Finance and"
            " MUST NOT be decided by you" in flattened
        )
        assert "do not pass `+12` or `-12`" in flattened
        assert (
            "do not decide whether the corrected operation is an income or"
            " an expense" in flattened
        )

    def test_direction_preserved_example_documented(self) -> None:
        flattened = " ".join(BODY.split())
        assert (
            "the last operation is `-10 Инфраструктура Сервер B` and the user says"
            " «Исправь последнюю сумму на 12»" in flattened
        )
        assert "remains an expense and becomes `-12 Инфраструктура Сервер B`" in flattened
        assert "the direction is never silently flipped" in flattened

    def test_soft_delete_semantics_documented(self) -> None:
        flattened = " ".join(BODY.split())
        assert "Deletion is a soft delete" in flattened
        assert "Repeated deletion of the same ID is idempotent" in flattened
        assert "no hard delete and no restore" in flattened

    def test_direction_change_request_not_emulated(self) -> None:
        flattened = " ".join(BODY.split())
        assert "The current v1 correction supports the amount only" in flattened
        assert (
            "do not emulate a direction change by deleting and re-creating"
            " the operation" in flattened
        )
        assert (
            "do not fabricate Telegram provenance for a replacement"
            " transaction" in flattened
        )
        assert "ask whether they want the amount changed instead" in flattened

    def test_success_only_after_cli_json(self) -> None:
        flattened = " ".join(BODY.split())
        assert (
            'edit: `{"disposition": "UPDATED", "transaction_id": "15"}`'
            in flattened
        )
        assert (
            'delete: `{"disposition": "DELETED", "transaction_id": "15"}`'
            in flattened
        )
        assert (
            "You may say «исправлено» or «удалено» ONLY after that JSON"
            " result is actually returned by the CLI" in flattened
        )

    def test_failure_can_never_become_success_wording(self) -> None:
        flattened = " ".join(BODY.split())
        assert "never claim that the ledger state changed" in flattened
        assert (
            "never answer «удалено», «исправлено», «готово», «сохранено»,"
            " or any equivalent success wording" in flattened
        )

    def test_mutation_no_false_success_rule_linked(self) -> None:
        flattened = " ".join(BODY.split())
        assert (
            "model interpretation alone is never a success confirmation"
            in flattened
        )

    def test_read_choose_mutate_flow_documented(self) -> None:
        flattened = " ".join(BODY.split())
        assert "The natural correction flow is read -> choose -> mutate" in (
            flattened
        )
        assert (
            "show a transaction list (for example"
            " `<cli-path> transactions recent`)" in flattened
        )
        assert "run the exact mutation command for that ID" in flattened

    def test_mutation_chain_documented(self) -> None:
        flattened = " ".join(BODY.split())
        for step in (
            "Telegram Finance topic",
            "blackcat-finance skill",
            "terminal",
            "hermes-finance transaction",
            "Finance core",
            "SQLite",
        ):
            assert step in flattened, step

    def test_mutation_goes_only_through_the_cli(self) -> None:
        flattened = " ".join(BODY.split())
        assert (
            "Every mutation goes through the `hermes-finance` CLI exactly"
            " like the reads above; there is no direct database access,"
            " no SQL, and no other mutation path" in flattened
        )

    def test_fast_create_ownership_not_blurred(self) -> None:
        """Fast CREATE stays with the native plugin; only corrections and
        deletion of already stored operations belong to this skill."""
        flattened = " ".join(BODY.split())
        assert (
            "Fast transaction persistence (creating new operations from"
            " live Telegram messages) is a live native capability owned by"
            " the separate `blackcat-finance-fastpath` Hermes plugin, not"
            " by this skill" in flattened
        )
        assert (
            "Corrections and deletion of already stored operations, in"
            " contrast, are owned by this skill through the mutation"
            " commands below" in flattened
        )

    def test_hard_prohibition_on_false_mutation_success(self) -> None:
        flattened = " ".join(BODY.split())
        assert (
            "Never report a correction or deletion as successful without"
            " the finance CLI's deterministic JSON confirmation" in flattened
        )

    def test_no_hard_delete_instruction(self) -> None:
        flattened = " ".join(BODY.split())
        assert "DELETE FROM" not in SKILL_TEXT
        assert "hard delete" in flattened


class TestFilteredMonthlySummaries:
    """G2: the skill teaches the filtered monthly summary commands exactly.

    The summary card returns four authoritative metrics (income, expense,
    net, count). The model may quote one of them but must never calculate
    another number from them, never merge ambiguous labels, and maps
    relative months to the relative summary commands instead of deriving
    any month itself.
    """

    def test_summary_command_documented(self) -> None:
        flattened = " ".join(BODY.split())
        assert (
            "`<cli-path> summary month --year YYYY --month M"
            ' --category "CATEGORY"`' in flattened
        )
        assert (
            "`<cli-path> summary month --year YYYY --month M"
            ' --category "CATEGORY" --source "SOURCE"`' in flattened
        )

    def test_summary_examples_documented(self) -> None:
        flattened = " ".join(BODY.split())
        assert (
            '`<cli-path> summary month --year 2026 --month 9'
            ' --category "Работа"`' in flattened
        )
        assert (
            '`<cli-path> summary month --year 2026 --month 9'
            ' --category "Работа" --source "Проект A"`' in flattened
        )

    def test_category_required_and_source_category_local(self) -> None:
        flattened = " ".join(BODY.split())
        assert "`--category` is always required" in flattened
        assert "`--source` always means a source inside that category" in flattened
        assert "no source-only query" in flattened
        assert "never merged into one number" in flattened

    def test_four_authoritative_metrics_documented(self) -> None:
        flattened = " ".join(BODY.split())
        assert (
            "four authoritative metrics: income, expense, net, and"
            " operation count" in flattened
        )
        assert "quoting the corresponding authoritative value" in flattened

    def test_model_prohibited_from_money_arithmetic(self) -> None:
        flattened = " ".join(BODY.split())
        assert "MUST NOT be altered or extended by model arithmetic" in flattened
        assert "do not sum, subtract, merge, or otherwise calculate" in flattened
        assert "never recompute, sum, or merge money figures yourself" in flattened
        assert (
            "do not combine values of several categories or sources into"
            " one number" in flattened
        )

    def test_no_data_semantics_documented(self) -> None:
        flattened = " ".join(BODY.split())
        assert "📭 Данных нет." in flattened
        assert "does not exist in that month" in flattened
        assert "never present a zero total for a label that does not exist" in (
            flattened
        )

    def test_ambiguous_label_must_not_be_merged(self) -> None:
        flattened = " ".join(BODY.split())
        assert "If the label is uncertain" in flattened
        assert "do not merge them" in flattened
        assert "do not add their money values" in flattened
        assert "do not guess silently" in flattened
        assert (
            "run `<cli-path> month --year YYYY --month M` to inspect the"
            " available exact category and source labels" in flattened
        )
        assert "or ask the user which category or source they mean" in flattened

    def test_natural_language_to_exact_label_mapping_documented(self) -> None:
        flattened = " ".join(BODY.split())
        assert "may map clear natural-language wording to one exact" in flattened
        assert "«на работе» clearly refers to the persisted category" in flattened

    def test_income_by_category_intent_documented(self) -> None:
        assert (
            "«Сколько заработал на Работе за сентябрь 2026?» ->"
            ' `<cli-path> summary month --year 2026 --month 9'
            ' --category "Работа"`' in BODY
        )

    def test_expense_by_category_intent_documented(self) -> None:
        assert (
            "«Сколько потратил на Инфраструктуру за сентябрь 2026?» ->"
            ' `<cli-path> summary month --year 2026 --month 9'
            ' --category "Инфраструктура"`' in BODY
        )

    def test_net_by_category_intent_documented(self) -> None:
        assert (
            "«Какой итог по Работе за сентябрь 2026?» ->"
            ' `<cli-path> summary month --year 2026 --month 9'
            ' --category "Работа"`' in BODY
        )

    def test_category_and_source_intent_documented(self) -> None:
        assert (
            "«Сколько было по Проекту A в категории Работа за сентябрь 2026?» ->"
            ' `<cli-path> summary month --year 2026 --month 9'
            ' --category "Работа" --source "Проект A"`' in BODY
        )

    def test_missing_year_or_month_asks_instead_of_guessing(self) -> None:
        flattened = " ".join(BODY.split())
        assert (
            "If the user did not state the year or the month explicitly,"
            " ask for the missing part instead of guessing it" in flattened
        )

    def test_summary_chain_documented(self) -> None:
        flattened = " ".join(BODY.split())
        for step in (
            "Telegram Finance topic",
            "blackcat-finance skill",
            "terminal",
            "hermes-finance summary month",
            "Finance core",
            "SQLite",
        ):
            assert step in flattened, step

    def test_summary_month_relative_form_documented(self) -> None:
        """G3: `summary month` has relative alternatives, not a refusal."""
        flattened = " ".join(BODY.split())
        assert (
            "`<cli-path> summary month --relative current"
            ' --timezone "<business-timezone>" --category "CATEGORY"`' in flattened
        )
        assert (
            "`<cli-path> summary month --relative previous"
            ' --timezone "<business-timezone>" --category "CATEGORY"`' in flattened
        )


class TestRelativePeriodsSupported:
    """G3: the four planned relative periods map to canonical CLI commands.

    «сегодня», «вчера», «этот месяц», and «прошлый месяц» map to the
    relative flags of the existing commands, always carrying the
    configured ``<business-timezone>``. The model never derives a date
    itself: the finance CLI resolves the exact period and renders the
    real resolved date or month.
    """

    def test_relative_period_section_present(self) -> None:
        assert "## Relative periods" in BODY

    def test_old_relative_period_prohibition_absent(self) -> None:
        assert "## Relative periods are not supported" not in BODY
        flattened = " ".join(BODY.split())
        assert "Relative periods are not supported" not in flattened
        assert "MUST NOT invent a date for a relative request" not in flattened
        assert "ask the user for the exact calendar date" not in flattened

    def test_today_mappings(self) -> None:
        assert (
            "«Что записано сегодня?» ->"
            ' `<cli-path> transactions date --relative today'
            ' --timezone "<business-timezone>"`' in BODY
        )
        assert (
            "«Покажи операции за сегодня» ->"
            ' `<cli-path> transactions date --relative today'
            ' --timezone "<business-timezone>"`' in BODY
        )

    def test_yesterday_mappings(self) -> None:
        assert (
            "«Что записано вчера?» ->"
            ' `<cli-path> transactions date --relative yesterday'
            ' --timezone "<business-timezone>"`' in BODY
        )
        assert (
            "«Покажи операции за вчера» ->"
            ' `<cli-path> transactions date --relative yesterday'
            ' --timezone "<business-timezone>"`' in BODY
        )

    def test_current_month_transaction_mapping(self) -> None:
        assert (
            "«Покажи операции за этот месяц» ->"
            ' `<cli-path> transactions month --relative current'
            ' --timezone "<business-timezone>"`' in BODY
        )

    def test_previous_month_transaction_mapping(self) -> None:
        assert (
            "«Покажи операции за прошлый месяц» ->"
            ' `<cli-path> transactions month --relative previous'
            ' --timezone "<business-timezone>"`' in BODY
        )

    def test_current_month_report_mapping(self) -> None:
        assert (
            "«Покажи отчёт за этот месяц» ->"
            ' `<cli-path> month --relative current'
            ' --timezone "<business-timezone>"`' in BODY
        )

    def test_previous_month_report_mapping(self) -> None:
        assert (
            "«Покажи отчёт за прошлый месяц» ->"
            ' `<cli-path> month --relative previous'
            ' --timezone "<business-timezone>"`' in BODY
        )

    def test_current_month_filtered_summary_mapping(self) -> None:
        assert (
            "«Сколько заработал на Работе в этом месяце?» ->"
            ' `<cli-path> summary month --relative current'
            ' --timezone "<business-timezone>" --category "Работа"`' in BODY
        )
        assert (
            "«Какой итог по Работе в этом месяце?» ->"
            ' `<cli-path> summary month --relative current'
            ' --timezone "<business-timezone>" --category "Работа"`' in BODY
        )

    def test_previous_month_filtered_summary_mapping(self) -> None:
        assert (
            "«Сколько потратил на Инфраструктуру в прошлом месяце?» ->"
            ' `<cli-path> summary month --relative previous'
            ' --timezone "<business-timezone>" --category "Инфраструктура"`' in BODY
        )

    def test_relative_command_forms_documented(self) -> None:
        flattened = " ".join(BODY.split())
        assert (
            "`<cli-path> transactions date --relative today"
            ' --timezone "<business-timezone>"`' in flattened
        )
        assert (
            "`<cli-path> transactions date --relative yesterday"
            ' --timezone "<business-timezone>"`' in flattened
        )
        assert (
            "`<cli-path> transactions month --relative current"
            ' --timezone "<business-timezone>"`' in flattened
        )
        assert (
            "`<cli-path> transactions month --relative previous"
            ' --timezone "<business-timezone>"`' in flattened
        )
        assert (
            "`<cli-path> month --relative current"
            ' --timezone "<business-timezone>"`' in flattened
        )
        assert (
            "`<cli-path> month --relative previous"
            ' --timezone "<business-timezone>"`' in flattened
        )
        assert (
            "`<cli-path> summary month --relative current"
            ' --timezone "<business-timezone>" --category "CATEGORY"`' in flattened
        )
        assert (
            "`<cli-path> summary month --relative previous"
            ' --timezone "<business-timezone>" --category "CATEGORY"`' in flattened
        )

    def test_resolved_period_rendered_not_vague_wording(self) -> None:
        flattened = " ".join(BODY.split())
        assert "renders the real resolved date or month" in flattened
        assert "never vague wording" in flattened

    def test_model_prohibited_from_date_arithmetic(self) -> None:
        flattened = " ".join(BODY.split())
        assert "MUST NOT derive any calendar date itself" in flattened
        assert (
            "do not inspect the current date to convert «сегодня» or"
            " «вчера» into an exact date" in flattened
        )
        assert "do not calculate yesterday or the previous month" in flattened
        assert "do not infer a timezone" in flattened
        assert (
            "do not transform relative wording into explicit year/month"
            " values yourself" in flattened
        )
        assert "This applies even if the current date is known" in flattened

    def test_missing_timezone_fail_closed(self) -> None:
        flattened = " ".join(BODY.split())
        assert "do not guess a timezone" in flattened
        assert "do not use the system timezone" in flattened
        assert "do not fall back to UTC" in flattened
        assert (
            "tell the operator that the Finance business timezone requires"
            " configuration" in flattened
        )

    def test_explicit_commands_usable_without_timezone_config(self) -> None:
        flattened = " ".join(BODY.split())
        assert "Explicit-period commands remain usable without this value" in flattened

    def test_business_timezone_placeholder_defined(self) -> None:
        flattened = " ".join(BODY.split())
        assert "`<business-timezone>` means the current configured value" in flattened
        assert "blackcat_finance.business_timezone" in flattened

    def test_unsupported_relative_wording_still_refused(self) -> None:
        flattened = " ".join(BODY.split())
        assert "«позавчера»" in flattened
        assert "«последние 7 дней»" in flattened
        assert "is NOT supported" in flattened
        assert "do not invent a command for it" in flattened
        assert "do not compute a date for it" in flattened


class TestFailClosedCliConfig:
    def test_fail_closed_behavior_documented(self) -> None:
        flattened = " ".join(BODY.split())
        assert "Always use the injected `blackcat_finance.cli_path` value" in flattened
        assert "If it is absent or unconfigured:" in flattened

    def test_no_guessed_path_when_unconfigured(self) -> None:
        assert "do not guess a path" in BODY

    def test_finance_not_executed_when_unconfigured(self) -> None:
        assert "do not execute Finance" in BODY

    def test_operator_informed_of_missing_configuration(self) -> None:
        assert "tell the operator that the Finance CLI path needs configuration" in BODY


class TestFastInputContract:
    def test_fast_grammar_documented(self) -> None:
        assert "<sign><amount>[USDT] <category> <source> [| <comment>]" in BODY

    def test_fast_grammar_examples_exist(self) -> None:
        for example in (
            "+25 Работа Проект A",
            "+25 Работа Проект B",
            "+25 Сервисы Подписка",
            "-10 Инфраструктура Хостинг",
            "-1 Инфраструктура Сервер A",
            "-10 Инфраструктура Сервер B",
        ):
            assert example in BODY

    def test_live_ingest_not_instructed_as_skill_action(self) -> None:
        ingest_paragraphs = [
            paragraph
            for paragraph in BODY_PARAGRAPHS
            if "ingest-telegram" in paragraph
        ]
        assert ingest_paragraphs, "ingest-telegram must be mentioned explicitly"
        for paragraph in ingest_paragraphs:
            if _is_indented_diagram(paragraph):
                # The architecture diagram of the native plugin path is a
                # factual description, not a skill instruction.
                continue
            assert "MUST NOT" in paragraph
        # The skill-side prohibition is stated explicitly and verbatim.
        flattened = " ".join(BODY.split())
        assert "MUST NOT run `hermes-finance ingest-telegram`" in flattened

    def test_fabricated_telegram_provenance_prohibited(self) -> None:
        assert "NEVER fabricate" in BODY
        for token in (
            "update_id",
            "message_id",
            "message_thread_id",
            "chat_id",
            "HERMES_SESSION_ID",
        ):
            assert token in BODY

    def test_no_false_success(self) -> None:
        assert "Never answer «Сохранено»" in BODY
        assert "actually returns success" in BODY


class TestNativeFastpathAcknowledgement:
    """F4R2: the skill must describe the live native fast path accurately.

    The native ``blackcat-finance-fastpath`` plugin (stage F5) is live in
    production: fast lines in the Finance Telegram topic ARE persisted. The
    skill must acknowledge that instead of telling the user that automatic
    saving is unavailable, while still never performing the ingest itself.
    """

    def test_native_fastpath_plugin_named(self) -> None:
        assert "blackcat-finance-fastpath" in BODY

    def test_fastpath_identified_as_separate_persistence_owner(self) -> None:
        flattened = " ".join(BODY.split())
        assert "owned by the separate `blackcat-finance-fastpath` Hermes plugin" in flattened

    def test_fast_lines_supported_in_finance_topic(self) -> None:
        flattened = " ".join(BODY.split())
        assert "ARE supported in the Finance Telegram topic" in flattened

    def test_fast_transaction_chain_documented(self) -> None:
        flattened = " ".join(BODY.split())
        for step in (
            "Telegram update",
            "native blackcat-finance-fastpath plugin",
            "hermes-finance ingest-telegram",
            "Finance core",
            "SQLite",
        ):
            assert step in flattened, step

    def test_monthly_report_chain_documented(self) -> None:
        flattened = " ".join(BODY.split())
        for step in (
            "Telegram Finance topic",
            "blackcat-finance skill",
            "terminal",
            "hermes-finance month",
            "Finance core",
            "SQLite",
        ):
            assert step in flattened, step

    def test_skill_must_not_run_ingest_telegram(self) -> None:
        flattened = " ".join(BODY.split())
        assert "MUST NOT run `hermes-finance ingest-telegram`" in flattened
        assert "MUST NOT persist it itself" in flattened

    def test_provenance_owned_by_native_fastpath(self) -> None:
        flattened = " ".join(BODY.split())
        assert (
            "the native `blackcat-finance-fastpath` plugin receives them "
            "from the real Telegram update" in flattened
        )

    def test_no_false_success_without_fastpath_confirmation(self) -> None:
        flattened = " ".join(BODY.split())
        assert "without confirmation from the native Finance fast-path" in flattened
        assert "not confirmed by the native Finance fast-path" in flattened
        assert "no save confirmation can be given" in flattened

    def test_fail_closed_does_not_claim_general_unavailability(self) -> None:
        flattened = " ".join(BODY.split())
        assert (
            "do not say that fast persistence is generally unavailable"
            in flattened
        )


class TestStalePreF5LanguageAbsent:
    """F4R2 regression guard: stale pre-F5 wording must not return.

    Before F5 went live, the skill said the native fast path would arrive
    later and that fast-line persistence was unavailable. That made the
    model tell the user automatic saving was not possible. These pins keep
    every such obsolete statement out of the skill.
    """

    def test_stage_f5_future_wording_absent(self) -> None:
        assert "Stage F5 will provide" not in SKILL_TEXT
        assert "stage F5 will" not in SKILL_TEXT.lower()

    def test_until_stage_f5_wording_absent(self) -> None:
        assert "Until stage F5" not in SKILL_TEXT

    def test_until_then_wording_absent(self) -> None:
        assert "Until then" not in SKILL_TEXT

    def test_unavailable_persistence_claim_absent(self) -> None:
        lowered = SKILL_TEXT.lower()
        assert "persistence is not available" not in lowered
        assert "persistence is unavailable" not in lowered
        assert "saving is not available" not in lowered
        assert "недоступно" not in SKILL_TEXT

    def test_no_stage_f4_self_reference(self) -> None:
        assert "stage F4" not in SKILL_TEXT
        assert "(F4)" not in SKILL_TEXT


class TestSourceHygiene:
    def test_no_direct_sqlite_instructions(self) -> None:
        for paragraph in BODY_PARAGRAPHS:
            if "sqlite3" in paragraph:
                assert "Never" in paragraph
        for marker in FORBIDDEN_SQL_MARKERS:
            assert marker not in SKILL_TEXT

    def test_no_llm_money_calculation(self) -> None:
        assert "Never calculate exact money figures" in BODY

    def test_no_hermes_core_path_dependency(self) -> None:
        assert "hermes-agent" not in SKILL_TEXT.lower()
        assert "/home/" not in SKILL_TEXT

    def test_no_real_credentials_or_ids(self) -> None:
        assert not re.search(r"\d{7,}", SKILL_TEXT)
        assert not re.search(r"\d{5,10}:[A-Za-z0-9_-]{30,}", SKILL_TEXT)

    def test_no_http_service_or_systemd_architecture(self) -> None:
        lowered = SKILL_TEXT.lower()
        for marker in FORBIDDEN_INFRASTRUCTURE:
            assert marker not in lowered, marker
