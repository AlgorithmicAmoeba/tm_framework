from sqlalchemy import text
from database import get_engine, get_session, TestSession
from configuration import load_config_from_env

def test_get_engine(config):
    engine = get_engine(config.database)
    assert engine is not None

def test_get_session(config):
    session_class = get_session(config.database)
    assert session_class is not None
    session = session_class()
    assert session.bind is not None
    session.close()

def test_test_session(config):
    with TestSession(config.database) as session:
        # Ensure the session is active
        assert session.is_active

        # Execute a simple query to verify the session works
        result = session.execute(text("SELECT 1"))
        assert result.scalar() == 1

        # Ensure the session is rolled back after exiting the context
        session.execute(text("CREATE TABLE test_table (id INTEGER)"))
        session.commit()

    # After the context, the table should not exist
    with TestSession(config.database) as session:
        result = session.execute(text("SELECT to_regclass('test_table')"))
        assert result.scalar() is None
