import dataclasses
import json
import os

@dataclasses.dataclass
class Config:
    database: "DatabaseConfig"


@dataclasses.dataclass
class DatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    db_name: str

    def get_engine_string(self):
        res = f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}"
        return res


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw_config = json.load(f)

    database_config = DatabaseConfig(
        **raw_config["database"]
    )

    config = Config(database=database_config)
    
    return config


def load_config_from_env() -> Config:
    CONFIG_PATH = os.getenv("TM_FRAMEWORK_CONFIG_PATH")
    return load_config(CONFIG_PATH)