"""Bounded provider-neutral current-weather reads."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol, cast


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
class WeatherForecastDay:
    date: str
    temperature_max_c: float | None
    temperature_min_c: float | None
    precipitation_probability_max: float | None
    sunrise: str | None
    sunset: str | None
    source: str


class WeatherForecastProvider(Protocol):
    def forecast(
        self, latitude: float, longitude: float, days: int = 3
    ) -> tuple[WeatherForecastDay, ...]: ...


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

    def forecast(
        self, latitude: float, longitude: float, days: int = 3
    ) -> tuple[WeatherForecastDay, ...]:
        if not 1 <= days <= 7:
            raise ValueError("weather forecast days must be between 1 and 7")
        parsed = urllib.parse.urlsplit(self.endpoint)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("weather endpoint must use HTTPS")
        query = urllib.parse.urlencode(
            {
                "latitude": f"{latitude:.6f}",
                "longitude": f"{longitude:.6f}",
                "daily": (
                    "temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max,sunrise,sunset"
                ),
                "forecast_days": days,
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
            raise RuntimeError("weather forecast provider unavailable") from exc
        daily = payload.get("daily") if isinstance(payload, dict) else None
        if not isinstance(daily, dict):
            raise RuntimeError("weather response has no daily forecast")
        dates = daily.get("time")
        maxes = daily.get("temperature_2m_max")
        mins = daily.get("temperature_2m_min")
        probabilities = daily.get("precipitation_probability_max")
        sunrises = daily.get("sunrise")
        sunsets = daily.get("sunset")
        arrays = (dates, maxes, mins, probabilities, sunrises, sunsets)
        if not all(isinstance(item, list) for item in arrays):
            raise RuntimeError("weather daily forecast is invalid")
        dates = cast(list[object], dates)
        maxes = cast(list[object], maxes)
        mins = cast(list[object], mins)
        probabilities = cast(list[object], probabilities)
        sunrises = cast(list[object], sunrises)
        sunsets = cast(list[object], sunsets)
        result: list[WeatherForecastDay] = []
        for index in range(min(days, len(dates))):
            date = dates[index]
            if not isinstance(date, str):
                raise RuntimeError("weather forecast date is invalid")
            result.append(
                WeatherForecastDay(
                    date[:20],
                    _optional_float(maxes[index]),
                    _optional_float(mins[index]),
                    _optional_float(probabilities[index]),
                    _optional_text(sunrises[index]),
                    _optional_text(sunsets[index]),
                    "open_meteo",
                )
            )
        return tuple(result)


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


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _optional_text(value: object) -> str | None:
    return value[:80] if isinstance(value, str) else None


@dataclass(frozen=True)
class FixtureWeatherForecastProvider:
    days: tuple[WeatherForecastDay, ...]

    def forecast(
        self, latitude: float, longitude: float, days: int = 3
    ) -> tuple[WeatherForecastDay, ...]:
        del latitude, longitude
        if not 1 <= days <= 7:
            raise ValueError("weather forecast days must be between 1 and 7")
        return self.days[:days]


def configured_weather_forecast_provider() -> WeatherForecastProvider:
    raw = os.environ.get("AEGIS_WEATHER_FORECAST_FIXTURE_JSON")
    if raw:
        try:
            value = json.loads(raw)
            if not isinstance(value, list):
                raise ValueError
            days = tuple(
                WeatherForecastDay(
                    str(item["date"]),
                    _optional_float(item.get("temperature_max_c")),
                    _optional_float(item.get("temperature_min_c")),
                    _optional_float(item.get("precipitation_probability_max")),
                    _optional_text(item.get("sunrise")),
                    _optional_text(item.get("sunset")),
                    "fixture_weather",
                )
                for item in value[:7]
                if isinstance(item, dict)
            )
            if not days:
                raise ValueError
            return FixtureWeatherForecastProvider(days)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("weather forecast fixture configuration is invalid") from exc
    return OpenMeteoProvider(os.environ.get("AEGIS_WEATHER_ENDPOINT", OpenMeteoProvider.endpoint))


def weather_forecast_evidence(days: tuple[WeatherForecastDay, ...]) -> list[dict[str, object]]:
    return [
        {
            "date": day.date,
            "temperature_max_c": day.temperature_max_c,
            "temperature_min_c": day.temperature_min_c,
            "precipitation_probability_max": day.precipitation_probability_max,
            "sunrise": day.sunrise,
            "sunset": day.sunset,
            "source": day.source,
        }
        for day in days
    ]


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
