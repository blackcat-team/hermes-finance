# Hermes Finance

[![CI](https://github.com/blackcat-team/hermes-finance/actions/workflows/ci.yml/badge.svg)](https://github.com/blackcat-team/hermes-finance/actions/workflows/ci.yml)

**Status: stable v1 — deployed and in daily personal use.**

Hermes Finance is a lightweight personal bookkeeping system for USDT-denominated
income and expenses. It records transactions from a dedicated Telegram Finance
topic, stores them in a single SQLite ledger, and produces deterministic
monthly reports, transaction lists, and filtered summaries — either directly
through its CLI or through natural-language conversations with the
[Hermes](integrations/hermes) agent.

## Architecture

Two entry paths share one external CLI and one ledger:

```text
Telegram Finance topic (one dedicated chat topic)
        |
        | fast path: persist a transaction in one line
        v
native Hermes fast-path plugin (blackcat-finance-fastpath)
        |  subprocess, argv only — never a shell
        v
external hermes-finance CLI  <---- natural-language path:
        |                        Hermes skill (blackcat-finance)
        |                        terminal tool -> same CLI
        v
deterministic finance core (src/hermes_finance, stdlib only)
        |
        v
SQLite ledger (single file, USDT amounts stored as exact decimal text)
```

- **Fast path**: a tiny native Hermes plugin matches only the exact configured
  Telegram chat and Finance topic thread, and only message text starting with
  a signed amount (`+3` / `-3`). It attaches the real Telegram provenance and
  delegates everything to `hermes-finance ingest-telegram`. The plugin owns no
  finance logic: no parsing, no SQLite access, no LLM decisions.
- **Natural-language path**: a Hermes skill maps requests such as
  «Покажи отчёт за этот месяц» to the same external CLI (`month`,
  `transactions`, `summary month`, `transaction`). Exact money calculation and
  persistence always belong to the finance core, never to the model.
- **Update-neutral boundary**: the repository contains no tracked Hermes core
  modifications, and the CLI never imports from the Hermes virtualenv — it
  runs entirely in its own Python 3.11 environment. Hermes integrations live
  as a standalone plugin and skill under `integrations/` and can be updated
  independently of the Hermes host.

## Engineering guarantees

- **Python 3.11, zero runtime dependencies** — the finance core is pure
  standard library (`sqlite3`, `decimal`, `argparse`, `zoneinfo`, ...).
- **Exact money handling** — every amount is a `decimal.Decimal` end to end;
  SQLite stores amounts as decimal text, so no value is ever rounded through
  binary floating point.
- **Deterministic aggregation and rendering** — reports and lists are pure
  projections of stored values: no recalculation, no locale machinery, no
  wall-clock access in the core; rendered output is exact-string tested.
- **Idempotent ingest** — Telegram message/update provenance is enforced by
  database constraints; duplicate delivery never creates a second
  transaction.
- **Soft delete only** — deletion hides an operation from reports and lists
  while the stored history is preserved; there is deliberately no hard delete.
- **Fail-closed integration** — missing or invalid plugin/skill configuration
  registers no Telegram handler and never guesses defaults.
- **Finance topic scoping** — the fast path reacts only to the exact
  configured chat and topic thread; every unrelated message falls through to
  ordinary Hermes routing.

## Capabilities

- create a transaction from a one-line fast entry
- aggregated monthly report (income / expense / net / count, per category and
  per source)
- transaction views: recent, one exact date, one exact month
- filtered monthly summaries (one category, or one category and its source)
- relative periods: today, yesterday, current month, previous month —
  resolved deterministically in an explicit business timezone
- amount-only correction of a stored transaction (direction is preserved)
- soft delete of a stored transaction (idempotent)

## Fast entry grammar

```text
<sign><amount>[USDT] <category> <source> [| <comment>]
```

`+` is income, `-` is expense; the amount is a clean decimal string and only
USDT is supported.

```text
+25 Работа Проект A
-10 Инфраструктура Хостинг | продление сервера
+25.5 USDT Работа Проект B
```

## Natural-language requests

Through the Hermes skill in the Finance topic:

```text
Покажи отчёт за этот месяц
Что записано сегодня?
Сколько заработал на Работе в этом месяце?
Исправь сумму #15 на 12
Удали #15
```

## Example monthly report

`hermes-finance month --year 2026 --month 8` prints (user-facing output is
deterministic Russian plain text). The example below uses synthetic data.

```text
💰 FINANCE | АВГУСТ 2026

📈 Доход: 75 USDT
📉 Расход: 25 USDT
⚖️ Итог: +50 USDT
🧾 Операций: 5

📂 Инфраструктура
Доход: 0 USDT
Расход: 15 USDT
Итог: -15 USDT
Операций: 2

🔹 Источники
• Сервер A
  доход 0 · расход 5 · итог -5 USDT · операций 1
• Хостинг
  доход 0 · расход 10 · итог -10 USDT · операций 1

📂 Работа
Доход: 75 USDT
Расход: 0 USDT
Итог: +75 USDT
Операций: 2

🔹 Источники
• Проект A
  доход 40 · расход 0 · итог +40 USDT · операций 1
• Проект B
  доход 35 · расход 0 · итог +35 USDT · операций 1

📂 Сервисы
Доход: 0 USDT
Расход: 10 USDT
Итог: -10 USDT
Операций: 1

🔹 Источники
• Подписка
  доход 0 · расход 10 · итог -10 USDT · операций 1
```

## CLI

The `hermes-finance` console script is the only mutation and query surface.
It reads the ledger location from the `HERMES_FINANCE_DB_PATH` environment
variable (no default path is invented) and exits with deterministic errors on
invalid input. Command families:

```text
hermes-finance ingest-telegram --text TEXT --chat-id ID --message-id ID \
    --update-id ID --thread-id ID --transaction-date YYYY-MM-DD \
    --received-at ISO-8601          # idempotent intake of one Finance topic message

hermes-finance month --year 2026 --month 8        # aggregated monthly report
hermes-finance month --relative current --timezone Europe/Moscow

hermes-finance transactions recent [--limit N]    # newest transactions (last 10 by default)
hermes-finance transactions date --date 2026-09-05
hermes-finance transactions date --relative today --timezone Europe/Moscow
hermes-finance transactions month --year 2026 --month 9

hermes-finance summary month --year 2026 --month 9 --category "Работа"
hermes-finance summary month --year 2026 --month 9 --category "Работа" --source "Проект A"
hermes-finance summary month --relative current --timezone Europe/Moscow --category "Работа"

hermes-finance transaction edit-amount --id 15 --amount 12   # amount-only correction
hermes-finance transaction edit-amount --last --amount 12
hermes-finance transaction delete --id 15                    # idempotent soft delete
hermes-finance transaction delete --last
```

Ingest and mutations print one deterministic JSON object on success, for
example `{"disposition": "UPDATED", "transaction_id": "15"}`; nothing is
printed on stdout when an operation fails.

## Hermes integration layout

```text
integrations/hermes/
├── plugins/blackcat-finance-fastpath/   # native Hermes Telegram plugin
│   ├── plugin.yaml                      # fail-closed config schema (no defaults)
│   └── __init__.py                      # topic-scoped handler -> CLI subprocess
└── skills/blackcat-finance/
    └── SKILL.md                         # natural-language skill instructions
```

The production plugin copy lives in user-owned Hermes extension storage
(`~/.hermes/plugins/`), the skill copy in `~/.hermes/skills/`; both are
deployed without modifying tracked Hermes core files.

## Data ownership and safety

- All data lives in **one SQLite file you own** — no cloud service, no
  telemetry, no network access anywhere in the finance core.
- Deletion is a **soft delete**: history is preserved, and repeated deletion
  of the same transaction is idempotent.
- Telegram provenance (chat, thread, message, update ids) is recorded from
  the real Telegram update and **never fabricated**; it is what makes ingest
  idempotent.
- The repository contains **no secrets**: no tokens, no real chat or thread
  ids, no database paths. `.env.example` documents variable names only.

## Repository structure

```text
.
├── .github/workflows/ci.yml        # CI: pytest, ruff, mypy, pip check (Python 3.11)
├── docs/DEVELOPMENT.md             # local setup and verification commands
├── integrations/hermes/            # Hermes plugin and skill (see above)
├── src/hermes_finance/
│   ├── cli.py                      # hermes-finance console script boundary
│   ├── parser.py                   # fast-entry grammar parser
│   ├── domain.py                   # value objects, exact Decimal amounts
│   ├── ledger.py                   # transaction creation
│   ├── ingest.py                   # idempotent message ingest
│   ├── repository.py               # SQLite row mapping
│   ├── storage.py                  # schema bootstrap and migrations
│   ├── operations.py               # read-side selection
│   ├── reporting.py                # monthly aggregation
│   ├── filtered_summary.py         # category/source selection
│   ├── rendering.py                # deterministic plain-text views
│   ├── corrections.py              # amount-only correction
│   ├── mutations.py                # soft delete
│   ├── periods.py                  # relative period resolution
│   ├── provenance.py               # Telegram provenance reference
│   ├── config.py                   # configuration value object
│   └── integration.py              # high-level facade composition
└── tests/                          # fully deterministic pytest suite
```

## Development and verification

```text
python3.11 -m venv .venv                    # create the environment
python -m pip install -e ".[dev]"           # install package + dev tools
```

(Windows setup with the `py` launcher is described in
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).)

Canonical checks (also run by CI on every push and pull request):

```text
python -m pytest -q
python -m ruff check .
python -m mypy .
python -m pip check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow and
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for the full development guide.
