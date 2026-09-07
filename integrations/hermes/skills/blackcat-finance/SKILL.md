---
name: blackcat-finance
description: "Personal bookkeeping through the external hermes-finance CLI. Monthly finance report, filtered monthly summaries, transaction lists, and transaction corrections/deletion over the Finance ledger."
metadata:
  hermes:
    requires_tools:
      - terminal
    config:
      - key: blackcat_finance.cli_path
        description: Absolute path to the standalone hermes-finance CLI executable.
        prompt: Provide the absolute path to the hermes-finance CLI executable.
      - key: blackcat_finance.business_timezone
        description: IANA business timezone name (Region/City form) the finance CLI uses to resolve relative periods such as today, yesterday, this month, and previous month.
        prompt: Provide the IANA business timezone name (Region/City form) used for relative finance periods.
required_environment_variables:
  - name: HERMES_FINANCE_DB_PATH
    prompt: Provide the file path of the finance SQLite database.
    required_for: Hermes Finance CLI database access
---

# blackcat-finance

## Purpose

This skill teaches Hermes to operate the user's personal bookkeeping through
the external `hermes-finance` CLI.

This skill owns exactly four natural-language read/mutation capabilities:

- monthly finance report (aggregated totals per month)
- filtered monthly summaries (one exact category, or one exact category and
  its source, of one exact month)
- transaction lists (individual stored operations)
- transaction corrections/deletion (amount-only correction and soft deletion
  of already stored operations)

Fast transaction persistence (creating new operations from live Telegram
messages) is a live native capability owned by the separate
`blackcat-finance-fastpath` Hermes plugin, not by this skill. Corrections
and deletion of already stored operations, in contrast, are owned by this
skill through the mutation commands below.

The skill is instructions and configuration only. Exact money calculation and
persistence belong to Hermes Finance, never to the model:

    Hermes
      | terminal/subprocess
      v
    hermes-finance CLI
      |
      v
    finance core -> SQLite

## Current architecture

Four production paths share one Finance SQLite ledger.

Fast transaction (native path; the plugin owns persistence, not this skill):

    Telegram update
      -> native blackcat-finance-fastpath plugin
      -> hermes-finance ingest-telegram
      -> Finance core
      -> SQLite

Natural-language report (this skill):

    Telegram Finance topic
      -> blackcat-finance skill
      -> terminal
      -> hermes-finance month
      -> Finance core
      -> SQLite

Natural-language transaction list (this skill):

    Telegram Finance topic
      -> blackcat-finance skill
      -> terminal
      -> hermes-finance transactions
      -> Finance core
      -> SQLite

Filtered monthly summary (this skill):

    Telegram Finance topic
      -> blackcat-finance skill
      -> terminal
      -> hermes-finance summary month
      -> Finance core
      -> SQLite

Natural-language correction/deletion (this skill):

    Telegram Finance topic
      -> blackcat-finance skill
      -> terminal
      -> hermes-finance transaction
      -> Finance core
      -> SQLite

Fast transaction lines such as `+25 Работа Проект A` and
`-10 Инфраструктура Хостинг` ARE supported in the Finance Telegram topic: the
native `blackcat-finance-fastpath` plugin matches them, attaches the real
Telegram provenance, and persists them through
`hermes-finance ingest-telegram`. This skill MUST NOT run
`hermes-finance ingest-telegram` for a live Telegram message, because it
does not own native Telegram provenance.

## Required tool

This skill requires the Hermes terminal tool, declared in the frontmatter
under `metadata.hermes.requires_tools`. All finance work is performed by
executing the external CLI through the terminal tool. There is no custom
Python Hermes integration and no in-process import of Hermes Finance code.

## CLI path configuration

The CLI executable location is the non-secret skill config value:

    blackcat_finance.cli_path

It is declared in the frontmatter under `metadata.hermes.config` and must
hold the absolute path to the standalone `hermes-finance` CLI executable.
There is deliberately no default value: no production checkout path is
hard-coded.

Always use the injected `blackcat_finance.cli_path` value. If it is absent
or unconfigured:

- do not guess a path
- do not execute Finance
- tell the operator that the Finance CLI path needs configuration

The CLI runs in its own environment and does not depend on the Hermes
virtualenv.

In the procedures below, `<cli-path>` means the current configured value of
`blackcat_finance.cli_path`.

## Business timezone configuration

Relative Finance periods are defined in the configured business timezone,
the separate non-secret skill config value:

    blackcat_finance.business_timezone

It is declared in the frontmatter under `metadata.hermes.config` and must
hold an IANA timezone name in Region/City form. There is deliberately no
default value: no production timezone is hard-coded, and relative periods
are never resolved in the server-local timezone or in UTC by accident.

In the procedures below, `<business-timezone>` means the current configured
value of `blackcat_finance.business_timezone`. It is passed to the CLI
through `--timezone` exactly as configured.

If a relative request arrives and this value is absent or unconfigured:

- do not guess a timezone
- do not use the system timezone
- do not fall back to UTC
- tell the operator that the Finance business timezone requires
  configuration

Explicit-period commands remain usable without this value.

## Database environment

The external CLI requires the `HERMES_FINANCE_DB_PATH` environment
variable, declared as a top-level `required_environment_variables` entry in
the frontmatter. Hermes must resolve that requirement and pass the value to
terminal subprocesses when this skill executes the CLI. Never store an
actual path or any secret value inside this skill.

## Monthly finance report

For an explicitly requested month and year, execute through the terminal
tool:

    <cli-path> month --year YYYY --month M

Example: for August 2026, run `<cli-path> month --year 2026 --month 8`.

For the supported relative month requests, execute instead:

- this month: `<cli-path> month --relative current --timezone "<business-timezone>"`
- previous month: `<cli-path> month --relative previous --timezone "<business-timezone>"`

If the user did not state the year or the month explicitly, ask for the
missing part instead of guessing it. Relative wording such as «этот месяц»
or «прошлый месяц» is not a missing part: it is a supported relative
period, so use the relative commands above.

The command prints the authoritative deterministic Russian monthly report on
stdout. Present that stdout to the user. Minor conversational framing around
it is allowed, but the report itself is authoritative and MUST NOT be
altered:

- do not recalculate totals
- do not change amounts
- do not re-group categories or sources
- do not invent missing transactions

## Filtered monthly summaries

For a question about one exact category, or one exact category and its
source, in one explicitly stated month and year, execute through the
terminal tool:

- one category: `<cli-path> summary month --year YYYY --month M --category "CATEGORY"`
- one category and source: `<cli-path> summary month --year YYYY --month M --category "CATEGORY" --source "SOURCE"`

Examples:

- `<cli-path> summary month --year 2026 --month 9 --category "Работа"`
- `<cli-path> summary month --year 2026 --month 9 --category "Работа" --source "Проект A"`

For a question about one exact category, or one exact category and its
source, in the current or previous business month, execute instead:

- this month, one category: `<cli-path> summary month --relative current --timezone "<business-timezone>" --category "CATEGORY"`
- previous month, one category: `<cli-path> summary month --relative previous --timezone "<business-timezone>" --category "CATEGORY"`
- with a source, append ` --source "SOURCE"` exactly as for the explicit form

`--category` is always required; `--source` always means a source inside
that category. There is deliberately no source-only query: the same source
label under two different categories is never merged into one number.

If the user did not state the year or the month explicitly, ask for the
missing part instead of guessing it. Relative wording such as «в этом
месяце» or «в прошлом месяце» is not a missing part: it is a supported
relative period, so use the relative commands above.

The command prints the authoritative deterministic Russian summary card on
stdout with four authoritative metrics: income, expense, net, and operation
count. Present that stdout to the user. You may answer the user's specific
question (income, expense, net, or count) by quoting the corresponding
authoritative value from the card, but the card itself is authoritative and
MUST NOT be altered or extended by model arithmetic:

- do not sum, subtract, merge, or otherwise calculate any money figure
- do not combine values of several categories or sources into one number
- do not change amounts
- do not invent missing data

If the card shows `📭 Данных нет.`, the requested category or source label
does not exist in that month: report that plainly and never present a zero
total for a label that does not exist.

### Category and source labels

You own the intent understanding only: you may map clear natural-language
wording to one exact persisted category or source label, for example «на
работе» clearly refers to the persisted category `Работа`.

The finance CLI matches labels exactly. If the label is uncertain, or
several persisted labels could plausibly match the user's wording:

- do not merge them
- do not add their money values
- do not guess silently

Instead, run `<cli-path> month --year YYYY --month M` to inspect the
available exact category and source labels of that month, or ask the user
which category or source they mean.

## Transaction lists

For showing individual stored operations, execute through the terminal
tool:

- last operations: `<cli-path> transactions recent` (last 10 by default)
- last N operations: `<cli-path> transactions recent --limit N`
  (for example `<cli-path> transactions recent --limit 5`)
- one exact date: `<cli-path> transactions date --date YYYY-MM-DD`
  (for example `<cli-path> transactions date --date 2026-09-05`)
- today: `<cli-path> transactions date --relative today --timezone "<business-timezone>"`
- yesterday: `<cli-path> transactions date --relative yesterday --timezone "<business-timezone>"`
- one exact calendar month: `<cli-path> transactions month --year YYYY --month M`
  (for example `<cli-path> transactions month --year 2026 --month 9`)
- this month: `<cli-path> transactions month --relative current --timezone "<business-timezone>"`
- previous month: `<cli-path> transactions month --relative previous --timezone "<business-timezone>"`

If the user asks for the operations of a specific date or month but did not
state it explicitly, ask for the missing part instead of guessing it.
Relative wording such as «сегодня», «вчера», «этот месяц», or «прошлый
месяц» is not a missing part: it is a supported relative period, so use
the relative commands above.

The command prints the authoritative deterministic Russian transaction list
on stdout. Present that stdout to the user. Minor conversational framing
around it is allowed, but the list itself is authoritative and MUST NOT be
altered:

- do not recalculate amounts
- do not change or hide transaction identifiers (`#<id>` lines)
- do not re-sort operations
- do not invent missing transactions

A transaction list and the monthly report are distinct capabilities:

- «Покажи операции за сентябрь 2026» -> `<cli-path> transactions month --year 2026 --month 9`
  (individual operations of the month)
- «Покажи отчёт за сентябрь 2026» -> `<cli-path> month --year 2026 --month 9`
  (aggregated monthly report with totals)

## Transaction corrections and deletion

This skill also owns two deterministic mutation commands of the finance
CLI. Every mutation goes through the `hermes-finance` CLI exactly like
the reads above; there is no direct database access, no SQL, and no
other mutation path.

### Mutation targets

Two deterministic target forms exist and are mutually exclusive:

- the exact persisted transaction ID: `--id ID`
- the latest ACTIVE operation: `--last`

`--last` is resolved by the finance CLI itself with exactly the same
selection as `transactions recent --limit 1`. NEVER determine yourself
which transaction is "last": pass `--last` and let Finance resolve the
target.

Transaction IDs come only from Finance CLI output (the `#<id>` lines of
a transaction list) or from explicit user input:

- do not invent an ID
- do not guess an ID
- do not use a Telegram `message_id`, `update_id`, or any other Telegram
  provenance value as a transaction ID
- if the user refers to an operation without an ID and without "last"
  wording, show a transaction list first and let the user choose

### Delete

- «Удали последнюю операцию» -> `<cli-path> transaction delete --last`
- «Удали #15» -> `<cli-path> transaction delete --id 15`

Deletion is a soft delete: the operation disappears from reports and
lists, while the stored history is preserved. Repeated deletion of the
same ID is idempotent. There is deliberately no hard delete and no
restore.

### Edit amount

- «Исправь последнюю сумму на 12» -> `<cli-path> transaction edit-amount --last --amount 12`
- «Исправь сумму #15 на 12» -> `<cli-path> transaction edit-amount --id 15 --amount 12`

`--amount` is always the positive magnitude. The stored income/expense
direction is preserved by Finance and MUST NOT be decided by you:

- do not pass `+12` or `-12`
- do not decide whether the corrected operation is an income or an
  expense

Example: the last operation is `-10 Инфраструктура Сервер B` and the user says
«Исправь последнюю сумму на 12»; run
`<cli-path> transaction edit-amount --last --amount 12`. The operation
remains an expense and becomes `-12 Инфраструктура Сервер B`; the direction is
never silently flipped.

The current v1 correction supports the amount only. If the user
explicitly asks to change the income/expense direction (or the category,
source, comment, or date) of an operation:

- state briefly that the current correction supports the amount only
- do not emulate a direction change by deleting and re-creating the
  operation
- do not fabricate Telegram provenance for a replacement transaction
- ask whether they want the amount changed instead

### Read, choose, mutate

The natural correction flow is read -> choose -> mutate:

1. show a transaction list (for example `<cli-path> transactions recent`)
   so the user sees the `#<id>` lines
2. the user chooses an operation («Удали #15» or
   «Исправь сумму #15 на 12»)
3. run the exact mutation command for that ID

### Mutation success confirmation

A mutation counts as successful only when the finance CLI exits with
status 0 and prints its deterministic JSON result on stdout:

- edit: `{"disposition": "UPDATED", "transaction_id": "15"}`
- delete: `{"disposition": "DELETED", "transaction_id": "15"}`

You may say «исправлено» or «удалено» ONLY after that JSON result is
actually returned by the CLI.

If the CLI fails (a non-zero exit status, an error message, or unexpected
output):

- report the failure briefly
- never claim that the ledger state changed
- never answer «удалено», «исправлено», «готово», «сохранено», or any
  equivalent success wording

This is the same no-false-save rule as for the native fast path: model
interpretation alone is never a success confirmation.

## Relative periods

Four relative-period concepts are supported: «сегодня» (today),
«вчера» (yesterday), «этот месяц» (this month), and «прошлый месяц»
(previous month). The finance CLI resolves them deterministically in the
configured business timezone and renders the real resolved date or month
in its output titles, never vague wording like «СЕГОДНЯ» or «ТЕКУЩИЙ
МЕСЯЦ».

For a relative request, emit the canonical relative CLI command. The model
MUST NOT derive any calendar date itself:

- do not inspect the current date to convert «сегодня» or «вчера» into an
  exact date
- do not calculate yesterday or the previous month
- do not infer a timezone
- do not transform relative wording into explicit year/month values
  yourself

This applies even if the current date is known: the CLI captures one
current-time snapshot, resolves the exact period in `<business-timezone>`
through its pure period resolver, and feeds the result into the exact same
report, list, and summary paths as the explicit commands. The resolved
period shown in the CLI output is the authority.

Exact mappings:

- «Что записано сегодня?» -> `<cli-path> transactions date --relative today --timezone "<business-timezone>"`
- «Покажи операции за сегодня» -> `<cli-path> transactions date --relative today --timezone "<business-timezone>"`
- «Что записано вчера?» -> `<cli-path> transactions date --relative yesterday --timezone "<business-timezone>"`
- «Покажи операции за вчера» -> `<cli-path> transactions date --relative yesterday --timezone "<business-timezone>"`
- «Покажи операции за этот месяц» -> `<cli-path> transactions month --relative current --timezone "<business-timezone>"`
- «Покажи операции за прошлый месяц» -> `<cli-path> transactions month --relative previous --timezone "<business-timezone>"`
- «Покажи отчёт за этот месяц» -> `<cli-path> month --relative current --timezone "<business-timezone>"`
- «Покажи отчёт за прошлый месяц» -> `<cli-path> month --relative previous --timezone "<business-timezone>"`
- «Сколько заработал на Работе в этом месяце?» -> `<cli-path> summary month --relative current --timezone "<business-timezone>" --category "Работа"`
- «Сколько потратил на Инфраструктуру в прошлом месяце?» -> `<cli-path> summary month --relative previous --timezone "<business-timezone>" --category "Инфраструктура"`
- «Какой итог по Работе в этом месяце?» -> `<cli-path> summary month --relative current --timezone "<business-timezone>" --category "Работа"`

All other relative wording (for example «позавчера», «последние 7 дней»,
weeks, quarters, or arbitrary date ranges) is NOT supported: do not
invent a command for it, do not compute a date for it, and ask the user
for the exact date or month instead.

A relative command always carries `--timezone "<business-timezone>"`. If
that value is absent, see Business timezone configuration: do not guess,
ask the operator to configure it.

## Mapping user intent

Simple requests map to the commands, for example:

- «Покажи последние операции» -> `<cli-path> transactions recent`
- «Покажи последние 5 операций» -> `<cli-path> transactions recent --limit 5`
- «Покажи операции за 5 сентября 2026» -> `<cli-path> transactions date --date 2026-09-05`
- «Что записано 5 сентября 2026» -> `<cli-path> transactions date --date 2026-09-05`
- «Покажи операции за сентябрь 2026» -> `<cli-path> transactions month --year 2026 --month 9`
- «Покажи отчёт за август 2026» -> `<cli-path> month --year 2026 --month 8`
- «Сколько было за август 2026?» -> `<cli-path> month --year 2026 --month 8`
- «Сколько заработал на Работе за сентябрь 2026?» -> `<cli-path> summary month --year 2026 --month 9 --category "Работа"`
- «Сколько потратил на Инфраструктуру за сентябрь 2026?» -> `<cli-path> summary month --year 2026 --month 9 --category "Инфраструктура"`
- «Какой итог по Работе за сентябрь 2026?» -> `<cli-path> summary month --year 2026 --month 9 --category "Работа"`
- «Сколько было по Проекту A в категории Работа за сентябрь 2026?» -> `<cli-path> summary month --year 2026 --month 9 --category "Работа" --source "Проект A"`

The model only performs this intent understanding. Never build or use an NLP
parser: all deterministic finance work belongs to the finance CLI.

## Fast transaction grammar (recognition only)

Hermes Finance accepts fast transaction lines in the form:

    <sign><amount>[USDT] <category> <source> [| <comment>]

Accepted examples:

- `+25 Работа Проект A`
- `+25 Работа Проект B`
- `+25 Сервисы Подписка`
- `-10 Инфраструктура Хостинг`
- `-1 Инфраструктура Сервер A`
- `-10 Инфраструктура Сервер B`

Lines like these ARE supported in the Finance Telegram topic: the native
`blackcat-finance-fastpath` plugin persists them together with the real
Telegram provenance. You may recognize that a user message matches this
grammar, but this skill MUST NOT persist it itself and MUST NOT run
`hermes-finance ingest-telegram` for a live Telegram message.

## Telegram provenance is never fabricated

Persisting a Telegram transaction requires the real Telegram provenance
values owned by the native Telegram transport:

- `update_id`
- `message_id`
- `message_thread_id`
- `chat_id`

This skill does not own these values; the native
`blackcat-finance-fastpath` plugin receives them from the real Telegram
update. NEVER fabricate them and NEVER substitute any of the following for
Telegram provenance:

- `HERMES_SESSION_ID`
- constant numbers
- hashes
- timestamps
- random IDs
- inferred IDs

## No false success

If a fast transaction line reaches this ordinary skill/LLM path without
confirmation from the native Finance fast-path, fail closed:

- do not try to persist it yourself
- do not fabricate provenance
- do not claim that it was saved
- do not say that fast persistence is generally unavailable

Instead explain briefly that this message was not confirmed by the native
Finance fast-path and therefore no save confirmation can be given.
Never answer «Сохранено» or any equivalent success wording based on
model interpretation alone. A transaction counts as persisted only when
the deterministic Hermes Finance ingest path actually returns success,
and that confirmation belongs to the native fast-path plugin, not to
this skill.

## Hard prohibitions

- Never use the sqlite3 CLI, never run SQL, never inspect database tables,
  and never modify the database directly. Every operation goes through the
  `hermes-finance` CLI.
- Never report a correction or deletion as successful without the finance
  CLI's deterministic JSON confirmation; a failed or missing CLI result is
  a failure, never a success.
- Never calculate exact money figures by summing transactions yourself. For
  ledger and report questions, call the finance CLI when a supported command
  exists and treat its returned output as the authority; if no supported
  command exists, say so instead of computing. Never derive a new number
  from the metrics a command returns either: quote the authoritative value
  or show the card, never recompute, sum, or merge money figures yourself.
- Never create servers, services, or background infrastructure from this
  skill.

## Ownership and deployment

The source of truth for this skill lives in the Hermes-finance repository at
`integrations/hermes/skills/blackcat-finance/SKILL.md`. The production copy
lives in user-owned Hermes extension storage at
`~/.hermes/skills/blackcat-finance/`. The native
`blackcat-finance-fastpath` plugin is a separately deployed Hermes plugin,
not part of this skill. This skill never requires changes to
tracked Hermes core files.
