"""Bounded provider-neutral current-weather reads."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class WeatherReading:
    latitude: float
    longitude: float
    temperature_c: float
    apparent_temperature_c: float | None
    weather_code: int | None
    observed_at: str
    source: str


class WeatherProvider(Protocol):
    def current(self, latitude: float, longitude: float) -> WeatherReading: ...


@dataclass(frozen=True)
class FixtureWeatherProvider:
    reading: WeatherReading | None = None

    def current(self, latitude: float, longitude: float) -> WeatherReading:
        if self.reading is None:
            raise RuntimeError("weather fixture is not configured")
        return WeatherReading(
            latitude,
            longitude,
            self.reading.temperature_c,
            self.reading.apparent_temperature_c,
            self.reading.weather_code,
            self.reading.observed_at,
            self.reading.source,
        )


@dataclass(frozen=True)
class OpenMeteoProvider:
    endpoint: str = "https://api.open-meteo.com/v1/forecast"
    timeout_seconds: float = 5.0

    def current(self, latitude: float, longitude: float) -> WeatherReading:
        parsed = urllib.parse.urlsplit(self.endpoint)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("weather endpoint must use HTTPS")
        query = urllib.parse.urlencode(
            {
                "latitude": f"{latitude:.6f}",
                "longitude": f"{longitude:.6f}",
                "current": "temperature_2m,apparent_temperature,weather_code",
                "timezone": "UTC",
            }
        )
        request = urllib.request.Request(
            f"{self.endpoint}?{query}", headers={"Accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read(200_000))
        except Exception as exc:
            raise RuntimeError("weather provider unavailable") from exc
        current = payload.get("current") if isinstance(payload, dict) else None
        if not isinstance(current, dict):
            raise RuntimeError("weather response has no current conditions")
        temperature = current.get("temperature_2m")
        if not isinstance(temperature, (int, float)):
            raise RuntimeError("weather temperature is invalid")
        apparent = current.get("apparent_temperature")
        code = current.get("weather_code")
        return WeatherReading(
            latitude,
            longitude,
            float(temperature),
            float(apparent) if isinstance(apparent, (int, float)) else None,
            int(code) if isinstance(code, int) else None,
            str(current.get("time", ""))[:80],
            "open_meteo",
        )


def configured_weather_provider() -> WeatherProvider:
    raw = os.environ.get("AEGIS_WEATHER_FIXTURE_JSON")
    if raw:
        try:
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError
            return FixtureWeatherProvider(
                WeatherReading(
                    0.0,
                    0.0,
                    float(value["temperature_c"]),
                    (
                        float(value["apparent_temperature_c"])
                        if value.get("apparent_temperature_c") is not None
                        else None
                    ),
                    (int(value["weather_code"]) if value.get("weather_code") is not None else None),
                    str(value.get("observed_at", "fixture")),
                    "fixture_weather",
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("weather fixture configuration is invalid") from exc
    return OpenMeteoProvider(os.environ.get("AEGIS_WEATHER_ENDPOINT", OpenMeteoProvider.endpoint))


def weather_evidence(reading: WeatherReading) -> dict[str, object]:
    return {
        "source": reading.source,
        "latitude": reading.latitude,
        "longitude": reading.longitude,
        "temperature_c": reading.temperature_c,
        "apparent_temperature_c": reading.apparent_temperature_c,
        "weather_code": reading.weather_code,
        "observed_at": reading.observed_at,
    }
