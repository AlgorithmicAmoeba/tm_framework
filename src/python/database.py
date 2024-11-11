from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

import configuration as cfg

# global application scope.  create Session class, engine
__ENGINE = None
__Session = None


def get_engine(db_config: cfg.DatabaseConfig):
    global __ENGINE
    if __ENGINE is None:
        __ENGINE = create_engine(
            db_config.get_engine_string()
        )

    return __ENGINE


def get_session(db_config: cfg.DatabaseConfig):
    global __Session

    if __Session is None:
        __Session = sessionmaker(get_engine(db_config))

    return __Session


def get_test_session(db_config: cfg.DatabaseConfig):
    return TestSession(db_config)


class TestSession:
    """Taken almost verbatim from:
    https://docs.sqlalchemy.org/en/20/orm/session_transaction.html#joining-a-session-into-an-external-transaction-such-as-for-test-suites
    """

    def __init__(self, db_config: cfg.DatabaseConfig):
        self.db_config = db_config

    def __enter__(self):
        # connect to the database
        self.connection = get_engine(self.db_config).connect()

        # begin a non-ORM transaction
        self.trans = self.connection.begin()

        # bind an individual Session to the connection
        self.session = get_session(self.db_config)(bind=self.connection)

        #    optional     #

        # if the database supports SAVEPOINT (SQLite needs special
        # config for this to work), starting a savepoint
        # will allow tests to also use rollback within tests

        self.nested = self.connection.begin_nested()

        @event.listens_for(self.session, "after_transaction_end")
        def end_savepoint(session, transaction):
            if not self.nested.is_active:
                self.nested = self.connection.begin_nested()

        # ^^^ optional ^^^ #

        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()

        # rollback - everything that happened with the
        # Session above (including calls to commit())
        # is rolled back.
        self.trans.rollback()

        # return connection to the Engine
        self.connection.close()
