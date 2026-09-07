"""Tests for the immutable runtime configuration contract (stage A3).

All tests are deterministic: no network, no filesystem mutation (paths are
used as values only and are never created, opened, or resolved), no
wall-clock time, no randomness, and no external services.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any

import pytest

from hermes_finance import FinanceConfig

FIXED_POSITIVE_TZ = timezone(timedelta(hours=3))
FIXED_NEGATIVE_TZ = timezone(timedelta(hours=-5))


class _StringPathLike:
    """Minimal path-like object for testing ``os.PathLike`` acceptance."""

    def __fspath__(self) -> str:
        return "like/finance.sqlite"


class _BrokenTZInfo(tzinfo):
    """tzinfo that cannot provide a usable UTC offset."""

    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        return None

    def tzname(self, dt: datetime | None) -> str | None:
        return "broken"

    def dst(self, dt: datetime | None) -> timedelta | None:
        return None


class _ExplodingTZInfo(tzinfo):
    """tzinfo whose ``utcoffset`` raises an arbitrary exception."""

    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        raise RuntimeError("boom")

    def tzname(self, dt: datetime | None) -> str | None:
        return "exploding"

    def dst(self, dt: datetime | None) -> timedelta | None:
        return None


def make_config(**overrides: Any) -> FinanceConfig:
    """Build a valid FinanceConfig, applying keyword overrides (possibly invalid)."""
    params: dict[str, Any] = {
        "database_path": Path("data/finance.sqlite"),
        "business_timezone": FIXED_POSITIVE_TZ,
    }
    params.update(overrides)
    return FinanceConfig(**params)


class TestValidDatabasePath:
    def test_pathlib_path_accepted(self) -> None:
        config = make_config(database_path=Path("store/ledger.db"))
        assert config.database_path == Path("store/ledger.db")

    def test_clean_string_accepted(self) -> None:
        path: Any = "data/finance.sqlite"
        config = make_config(database_path=path)
        assert config.database_path == Path("data/finance.sqlite")

    def test_padded_string_is_trimmed(self) -> None:
        path: Any = "  data/finance.sqlite  "
        config = make_config(database_path=path)
        assert config.database_path == Path("data/finance.sqlite")

    def test_pathlike_accepted(self) -> None:
        path: Any = _StringPathLike()
        config = make_config(database_path=path)
        assert config.database_path == Path("like/finance.sqlite")

    def test_pathlike_pathlib_instance_accepted(self) -> None:
        path: Any = Path("nested/dir/finance.sqlite")
        config = make_config(database_path=path)
        assert config.database_path == Path("nested/dir/finance.sqlite")

    def test_stored_representation_is_pathlib_path(self) -> None:
        path: Any = "data/finance.sqlite"
        config = make_config(database_path=path)
        assert isinstance(config.database_path, Path)
        pathlike: Any = _StringPathLike()
        assert isinstance(make_config(database_path=pathlike).database_path, Path)

    def test_relative_path_stays_relative(self) -> None:
        config = make_config(database_path=Path("relative/finance.sqlite"))
        assert not config.database_path.is_absolute()
        string_path: Any = "relative/finance.sqlite"
        assert not make_config(database_path=string_path).database_path.is_absolute()

    def test_relative_path_is_not_resolved(self) -> None:
        relative = Path("relative/../finance.sqlite")
        config = make_config(database_path=relative)
        assert config.database_path == Path("relative/../finance.sqlite")
        # No implicit cwd resolution: the literal segments are preserved.
        cwd_resolved = (Path.cwd() / relative).resolve()
        assert config.database_path != cwd_resolved

    def test_nonexistent_path_accepted_without_existence_check(self) -> None:
        # A path that certainly does not exist is accepted; the config layer
        # must remain filesystem-side-effect free and never stat the path.
        config = make_config(database_path=Path("definitely/not/here/finance.sqlite"))
        assert not config.database_path.exists()

    def test_absolute_path_accepted(self) -> None:
        # Build an absolute path portably (drive/UNC anchor on Windows, "/" on POSIX).
        absolute = Path(Path.cwd().anchor) / "absolute" / "finance.sqlite"
        config = make_config(database_path=absolute)
        assert config.database_path == absolute
        assert config.database_path.is_absolute()

    def test_filename_has_no_required_suffix(self) -> None:
        config = make_config(database_path=Path("data/anything"))
        assert config.database_path == Path("data/anything")


class TestInvalidDatabasePath:
    def test_none_rejected(self) -> None:
        path: Any = None
        with pytest.raises(TypeError, match="database_path"):
            make_config(database_path=path)

    def test_bool_rejected(self) -> None:
        path: Any = True
        with pytest.raises(TypeError, match="bool"):
            make_config(database_path=path)
        path = False
        with pytest.raises(TypeError, match="bool"):
            make_config(database_path=path)

    def test_empty_string_rejected(self) -> None:
        path: Any = ""
        with pytest.raises(ValueError, match="database_path"):
            make_config(database_path=path)

    def test_whitespace_only_string_rejected(self) -> None:
        path: Any = "   "
        with pytest.raises(ValueError, match="database_path"):
            make_config(database_path=path)
        path = "\t \n"
        with pytest.raises(ValueError, match="database_path"):
            make_config(database_path=path)

    def test_numeric_rejected(self) -> None:
        path: Any = 42
        with pytest.raises(TypeError, match="unsupported type"):
            make_config(database_path=path)
        path = 3.14
        with pytest.raises(TypeError, match="unsupported type"):
            make_config(database_path=path)

    def test_arbitrary_object_rejected(self) -> None:
        path: Any = object()
        with pytest.raises(TypeError, match="unsupported type"):
            make_config(database_path=path)
        path = ["data/finance.sqlite"]
        with pytest.raises(TypeError, match="unsupported type"):
            make_config(database_path=path)

    def test_bytes_path_rejected(self) -> None:
        path: Any = b"data/finance.sqlite"
        with pytest.raises(TypeError, match="bytes"):
            make_config(database_path=path)


class TestValidBusinessTimezone:
    def test_datetime_utc_accepted(self) -> None:
        tz: Any = UTC
        config = make_config(business_timezone=tz)
        assert config.business_timezone is UTC

    def test_fixed_positive_offset_accepted(self) -> None:
        config = make_config(business_timezone=FIXED_POSITIVE_TZ)
        assert config.business_timezone is FIXED_POSITIVE_TZ

    def test_fixed_negative_offset_accepted(self) -> None:
        config = make_config(business_timezone=FIXED_NEGATIVE_TZ)
        assert config.business_timezone is FIXED_NEGATIVE_TZ

    def test_zero_offset_utc_timezone_accepted(self) -> None:
        tz: Any = timezone(timedelta(0))
        config = make_config(business_timezone=tz)
        assert config.business_timezone is tz

    def test_stored_timezone_is_supplied_object(self) -> None:
        supplied = timezone(timedelta(hours=9))
        config = make_config(business_timezone=supplied)
        assert config.business_timezone is supplied

    def test_fixed_offset_instances_are_tzinfo(self) -> None:
        # datetime.timezone is a datetime.tzinfo subclass.
        assert isinstance(FIXED_POSITIVE_TZ, tzinfo)


class TestInvalidBusinessTimezone:
    def test_none_rejected(self) -> None:
        tz: Any = None
        with pytest.raises(TypeError, match="business_timezone"):
            make_config(business_timezone=tz)

    def test_string_rejected(self) -> None:
        tz: Any = "Europe/Amsterdam"
        with pytest.raises(TypeError, match="tzinfo"):
            make_config(business_timezone=tz)
        tz = "UTC"
        with pytest.raises(TypeError, match="tzinfo"):
            make_config(business_timezone=tz)

    def test_int_rejected(self) -> None:
        tz: Any = 42
        with pytest.raises(TypeError, match="tzinfo"):
            make_config(business_timezone=tz)

    def test_arbitrary_object_rejected(self) -> None:
        tz: Any = object()
        with pytest.raises(TypeError, match="tzinfo"):
            make_config(business_timezone=tz)

    def test_broken_tzinfo_without_usable_offset_rejected(self) -> None:
        tz: Any = _BrokenTZInfo()
        with pytest.raises(ValueError, match="usable UTC offset"):
            make_config(business_timezone=tz)

    def test_exploding_tzinfo_raises_valueerror_not_runtimeerror(self) -> None:
        tz: Any = _ExplodingTZInfo()
        with pytest.raises(ValueError, match="usable UTC offset"):
            make_config(business_timezone=tz)

    def test_exploding_tzinfo_runtimeerror_does_not_escape(self) -> None:
        tz: Any = _ExplodingTZInfo()
        try:
            make_config(business_timezone=tz)
        except ValueError as error:
            # The public failure is a deterministic ValueError; the original
            # RuntimeError is chained for diagnostics, not the contract.
            assert not isinstance(error, RuntimeError)
            assert isinstance(error.__cause__, RuntimeError)
            assert str(error.__cause__) == "boom"
        else:
            pytest.fail("FinanceConfig should have raised ValueError")


class TestConfigSafety:
    def test_database_path_mutation_rejected(self) -> None:
        config = make_config()
        with pytest.raises(FrozenInstanceError):
            config.database_path = Path("other/finance.sqlite")  # type: ignore[misc]

    def test_business_timezone_mutation_rejected(self) -> None:
        config = make_config()
        with pytest.raises(FrozenInstanceError):
            config.business_timezone = FIXED_NEGATIVE_TZ  # type: ignore[misc]

    def test_equal_configs_are_equal(self) -> None:
        first = FinanceConfig(
            database_path=Path("data/finance.sqlite"),
            business_timezone=FIXED_POSITIVE_TZ,
        )
        second = FinanceConfig(
            database_path="data/finance.sqlite",  # type: ignore[arg-type]
            business_timezone=FIXED_POSITIVE_TZ,
        )
        assert first == second

    def test_no_module_level_default_instance(self) -> None:
        # The config module must not expose a singleton/default FinanceConfig
        # or any module-level config instance.
        import hermes_finance.config as config_module

        for name, value in vars(config_module).items():
            assert not isinstance(value, config_module.FinanceConfig), name
