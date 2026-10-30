"""Tests for rinchen.errors exception hierarchy."""

import pytest

from rinchen.errors import (
    ConfigError,
    DataError,
    DoctorCheckFailed,
    ParityError,
    RinchenError,
    UpstreamError,
)


@pytest.mark.parametrize(
    "exc_cls",
    [ConfigError, DataError, UpstreamError, ParityError, DoctorCheckFailed],
)
def test_each_domain_exception_is_a_rinchen_error(exc_cls):
    assert issubclass(exc_cls, RinchenError)
    assert issubclass(exc_cls, Exception)


def test_domain_exception_carries_message():
    with pytest.raises(ConfigError, match="bad config"):
        raise ConfigError("bad config")
