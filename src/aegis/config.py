"""Validated runtime configuration with secret-safe representations."""

from __future__ import annotations

import os

from pydantic import Field, SecretStr

from .contracts import StrictModel


class AegisConfig(StrictModel):
    environment: str = Field(default="development", pattern=r"^(development|test|production)$")
    database_url: SecretStr
    keycloak_url: str
    openfga_url: str
    openclaw_gateway_url: str
    ollama_url: str = "http://127.0.0.1:11434"
    allow_cloud_models: bool = False

    @classmethod
    def from_environment(cls) -> AegisConfig:
        required = {
            "database_url": "AEGIS_DATABASE_URL",
            "keycloak_url": "AEGIS_KEYCLOAK_URL",
            "openfga_url": "AEGIS_OPENFGA_URL",
            "openclaw_gateway_url": "AEGIS_OPENCLAW_GATEWAY_URL",
        }
        values: dict[str, str | bool] = {}
        missing: list[str] = []
        for field_name, env_name in required.items():
            value = os.environ.get(env_name)
            if not value:
                missing.append(env_name)
            else:
                values[field_name] = value
        if missing:
            raise ValueError(f"missing required AEGIS configuration: {', '.join(missing)}")
        values["environment"] = os.environ.get("AEGIS_ENVIRONMENT", "development")
        values["ollama_url"] = os.environ.get(
            "AEGIS_OLLAMA_URL", cls.model_fields["ollama_url"].default
        )
        values["allow_cloud_models"] = (
            os.environ.get("AEGIS_ALLOW_CLOUD_MODELS", "false").lower() == "true"
        )
        return cls.model_validate(values)
