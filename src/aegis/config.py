"""Validated runtime configuration with secret-safe representations."""

from __future__ import annotations

import os

from pydantic import Field, SecretStr, model_validator

from .contracts import StrictModel


class AegisConfig(StrictModel):
    environment: str = Field(default="development", pattern=r"^(development|test|production)$")
    database_url: SecretStr
    keycloak_url: str | None = None
    openfga_url: str | None = None
    openclaw_gateway_url: str | None = None
    ollama_url: str = "http://127.0.0.1:11434"
    allow_cloud_models: bool = False

    @model_validator(mode="after")
    def require_production_authority(self) -> AegisConfig:
        if self.environment == "production":
            missing = [
                name
                for name, value in (
                    ("keycloak_url", self.keycloak_url),
                    ("openfga_url", self.openfga_url),
                )
                if not value
            ]
            if missing:
                raise ValueError("production configuration requires " + ", ".join(missing))
        return self

    @classmethod
    def from_environment(cls) -> AegisConfig:
        required = {"database_url": "AEGIS_DATABASE_URL"}
        values: dict[str, str | bool] = {}
        for field_name, env_name in required.items():
            value = os.environ.get(env_name)
            if not value:
                raise ValueError(f"missing required AEGIS configuration: {env_name}")
            values[field_name] = value
        values["environment"] = os.environ.get("AEGIS_ENVIRONMENT", "development")
        for field_name, env_name in (
            ("keycloak_url", "AEGIS_KEYCLOAK_URL"),
            ("openfga_url", "AEGIS_OPENFGA_URL"),
            ("openclaw_gateway_url", "AEGIS_OPENCLAW_GATEWAY_URL"),
        ):
            if value := os.environ.get(env_name):
                values[field_name] = value
        values["ollama_url"] = os.environ.get(
            "AEGIS_OLLAMA_URL", cls.model_fields["ollama_url"].default
        )
        values["allow_cloud_models"] = (
            os.environ.get("AEGIS_ALLOW_CLOUD_MODELS", "false").lower() == "true"
        )
        return cls.model_validate(values)
