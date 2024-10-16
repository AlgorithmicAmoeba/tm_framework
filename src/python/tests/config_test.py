import pytest

from configuration import load_config_from_env

def test_load_config_from_env():
    
    config = load_config_from_env()

    assert config is not None

    assert config.database is not None

    assert config.database is not None

    assert config.database.host != ""
    assert config.database.port != 0
    assert config.database.user != ""
    assert config.database.password != ""
    assert config.database.db_name != ""

    assert config.database.get_engine_string() == f"postgresql+psycopg2://{config.database.user}:{config.database.password}@{config.database.host}:{config.database.port}/{config.database.db_name}"