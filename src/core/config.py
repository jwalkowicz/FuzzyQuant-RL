from __future__ import annotations

import os
from typing import Tuple, Type

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class YFinanceSettings(BaseModel):
    ticker: str = "SPY"
    period: str = "10y"
    interval: str = "1d"


class IndicatorSettings(BaseModel):
    rsi_window: int = 14


class TradingSettings(BaseModel):
    initial_balance: float = 10000.0
    action_size: int = 3
    observation_size: int = 9


class AgentSettings(BaseModel):
    learning_rate: float = 0.1
    initial_epsilon: float = 1.0
    epsilon_decay: float = 0.99
    final_epsilon: float = 0.1
    discount_factor: float = 0.95


class TrainingSettings(BaseModel):
    n_episodes: int = 100000


class TestingSettings(BaseModel): ...


class Settings(BaseSettings):
    yfinance: YFinanceSettings = YFinanceSettings()
    indicators: IndicatorSettings = IndicatorSettings()
    trading: TradingSettings = TradingSettings()
    agent: AgentSettings = AgentSettings()
    training: TrainingSettings = TrainingSettings()
    testing: TestingSettings = TestingSettings()

    model_config = SettingsConfigDict(
        env_file=".env", env_nested_delimiter="__", extra="ignore"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        active_env = os.getenv("ENV", "dev")
        yaml_path = f"config/{active_env}.yaml"

        sources = [init_settings, env_settings, dotenv_settings]

        if os.path.exists(yaml_path):
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path))

        return tuple(sources)


config = Settings()
