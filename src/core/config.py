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


class Settings(BaseSettings):
    yfinance: YFinanceSettings = YFinanceSettings()
    indicators: IndicatorSettings = IndicatorSettings()

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

        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path)

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            yaml_source,
        )


config = Settings()
