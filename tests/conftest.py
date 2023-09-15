"""Fixtures for use in multiple tests."""

import pytest


@pytest.fixture()
def _mocker_empty_config(mocker):
    """Read a mocked empty config file"""
    mocked_config = mocker.mock_open(
        read_data=''
    )
    mocker.patch('builtins.open', mocked_config)
