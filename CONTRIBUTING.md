# Contributing

## Requirements

- Python 3.11
- Git

## Setup

Create a virtual environment and install the package with its development
dependencies (detailed steps in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Workflow

- Use short-lived branches for implementation; keep the default branch
  stable.
- Keep changes focused and reviewable; do not mix unrelated refactors into
  feature or fix commits.
- Write commit subjects in the conventional style used by the history
  (`feat: ...`, `fix: ...`).

## Verification

Run the canonical checks from the repository root with the virtual
environment active before requesting a review — CI
(`.github/workflows/ci.yml`) runs the same suite on Python 3.11:

```text
python -m pytest -q
python -m ruff check .
python -m mypy .
python -m pip check
```

Do not invent checks that are not configured by the project; the tool
configuration in `pyproject.toml` is the single source of truth.

## Code expectations

- Runtime code is Python 3.11 standard library only; runtime dependencies
  stay empty.
- The source lives in `src/hermes_finance` (src layout) with tests in
  `tests/`; `pytest` resolves the src layout through `pyproject.toml`.
- `ruff` enforces the configured line length (100) and `py311` target;
  `mypy` runs in strict mode over `src` and `tests`.
- The only mypy exclusion is the standalone Hermes plugin directory
  `integrations/hermes/plugins/blackcat-finance-fastpath/`, which runs inside
  the Hermes host, not in this project's environment; every other
  present or future integration stays fully type-checked.
- Tests are fully deterministic: no network, no wall clock, no randomness.

## Secrets and data

- Keep secrets outside the repository; never commit tokens, real chat or
  thread ids, database paths, or personal finance data.
- Use `.env.example` only to document non-secret variable names.
