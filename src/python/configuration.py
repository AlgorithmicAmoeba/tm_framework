import dataclasses
import json
import os
import pathlib

@dataclasses.dataclass
class Config:
    database: "DatabaseConfig"
    openai: "OpenAIConfig"
    data_dir: "DataDirConfig"

    def get_data_path(self):
        return pathlib.Path(self.data_dir.data_dir)


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
    

@dataclasses.dataclass
class OpenAIConfig:
    api_key: str


@dataclasses.dataclass
class DataDirConfig:
    data_dir: str


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw_config = json.load(f)

    database_config = DatabaseConfig(
        **raw_config["database"]
    )

    openai_config = OpenAIConfig(
        **raw_config["openai"]
    )

    data_dir_config = DataDirConfig(
        **raw_config["data_dir"]
    )

    config = Config(
        database=database_config,
        openai=openai_config,
        data_dir=data_dir_config
    )
    
    return config


def load_config_from_env() -> Config:
    CONFIG_PATH = os.getenv("TM_FRAMEWORK_CONFIG_PATH")
    return load_config(CONFIG_PATH)