# Development

## Runtime

Python 3.11.

## Environment

Create the local virtual environment and install the package with its
development dependencies:

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Linux / macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Verification

With the virtual environment active, run the canonical supported checks from
the repository root:

```text
python -m pytest -q
python -m ruff check .
python -m mypy .
python -m pip check
```

An optional local syntax check is also supported:

```text
python -m compileall src
```

The same canonical checks run in GitHub Actions on Python 3.11 for every
push and pull request (`.github/workflows/ci.yml`).

Do not invent checks that are not configured by the project; the tool
configuration in `pyproject.toml` is the single source of truth.
