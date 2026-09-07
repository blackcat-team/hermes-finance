"""Immutable runtime configuration contract for Hermes Finance.

This module defines the frozen :class:`FinanceConfig` value object that
later ledger persistence and reporting layers will consume. It captures
exactly the runtime configuration concerns currently needed:

- ``database_path``: where the future persistence layer will store data
- ``business_timezone``: the timezone business-day boundaries are judged in

The module is deliberately free of side effects and environment access:

- no filesystem access: the database path is neither created, opened,
  resolved, nor checked for existence
- no environment variables, dotenv, config files, or Telegram/Hermes
  integration configuration is read
- no module-level mutable configuration state, no singleton, no default
  instance

Runtime initialisation (constructing the actual database, schema
bootstrap, and schema versioning) intentionally belongs to later stages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, time, tzinfo
from pathlib import Path
from typing import Final

__all__ = [
    "FinanceConfig",
]

# A fixed, deterministic probe datetime used only to verify that a supplied
# tzinfo can produce a usable UTC offset. The wall clock is never consulted.
# (Built naive on purpose: the supplied timezone is attached during probing.)
_TZ_PROBE: Final[datetime] = datetime.combine(date(2000, 1, 1), time(12, 0))


def _coerce_database_path(value: object) -> Path:
    """Normalise and validate the database path, returning a ``Path``.

    Accepted inputs:

    - :class:`pathlib.Path`
    - a non-empty ``str`` (surrounding whitespace is stripped)
    - any ``os.PathLike`` whose ``__fspath__`` yields a non-empty ``str``

    Rejected inputs:

    - ``None``
    - ``bool``
    - empty or whitespace-only strings
    - ``bytes`` paths and every other unsupported type

    The path is never resolved against the current working directory, is
    never checked for existence, and no directories or files are created.
    A relative path stays relative exactly as supplied.
    """
    if value is None:
        raise TypeError("database_path must not be None")
    if isinstance(value, bool):
        raise TypeError("database_path must not be a bool")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("database_path must not be an empty or whitespace-only string")
        return Path(text)
    if isinstance(value, os.PathLike):
        raw: object = os.fspath(value)
        if isinstance(raw, bytes):
            raise TypeError("database_path must not be a bytes path")
        if not isinstance(raw, str):
            raise TypeError("database_path must be string-like")
        text = raw.strip()
        if not text:
            raise ValueError("database_path must not be an empty or whitespace-only string")
        return Path(text)
    raise TypeError(f"database_path has unsupported type {type(value).__name__!r}")


def _validate_business_timezone(value: object) -> tzinfo:
    """Require a usable ``datetime.tzinfo`` instance.

    ``datetime.UTC``, fixed-offset :class:`datetime.timezone` instances,
    and later ``zoneinfo.ZoneInfo`` objects (constructed by deployment
    configuration) are all accepted. Strings are rejected: A3 must not
    depend on an installed IANA tz database.

    Usability is verified with a single deterministic fixed datetime probe
    (never the wall clock): a tzinfo whose ``utcoffset`` yields ``None``
    for the probe cannot support timezone-aware arithmetic and is
    rejected. Failures raised by the supplied tzinfo implementation
    itself while computing the offset (for example a ``RuntimeError``
    from a broken subclass) are converted into a deterministic
    ``ValueError``: the original cause is chained for diagnostics but is
    never allowed to escape as the public contract. The supplied
    timezone object is stored as-is and is never converted into another
    timezone.
    """
    if not isinstance(value, tzinfo):
        raise TypeError("business_timezone must be a datetime.tzinfo instance")
    probe = _TZ_PROBE.replace(tzinfo=value)
    try:
        offset = probe.utcoffset()
    except Exception as error:
        raise ValueError(
            "business_timezone failed to provide a usable UTC offset "
            f"for the probe datetime: {error!r}"
        ) from error
    if offset is None:
        raise ValueError("business_timezone must provide a usable UTC offset")
    return value


@dataclass(frozen=True, slots=True)
class FinanceConfig:
    """Immutable runtime configuration for Hermes Finance.

    Both fields are caller supplied and validated during construction.
    Because the dataclass is frozen, ordinary attribute assignment cannot
    silently change configuration after construction.

    ``database_path`` accepts a :class:`pathlib.Path`, a non-empty string,
    or a path-like object, and is always stored as a ``Path`` exactly as
    supplied (relative stays relative; no resolution, no existence check,
    no filesystem access).

    ``business_timezone`` accepts a ``datetime.tzinfo`` instance (for
    example ``datetime.UTC`` or a fixed-offset ``datetime.timezone``) that
    can provide a usable UTC offset for a deterministic probe datetime.
    """

    database_path: Path
    business_timezone: tzinfo

    def __post_init__(self) -> None:
        object.__setattr__(self, "database_path", _coerce_database_path(self.database_path))
        object.__setattr__(
            self, "business_timezone", _validate_business_timezone(self.business_timezone)
        )
