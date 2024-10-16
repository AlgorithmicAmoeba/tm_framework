import pytest

from configuration import load_config_from_env
from database import TestSession


@pytest.fixture(scope="package")
def config():
    # Load the configuration from the environment
    return load_config_from_env()


@pytest.fixture(scope="package")
def test_session():
    return TestSession(config.database)