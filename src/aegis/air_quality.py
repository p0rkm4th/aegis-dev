"""Bounded public air-quality reads."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class AirQualityReading:
    latitude: float
    longitude: float
    us_aqi: float | None
    pm2_5: float | None
    observed_at: str
    source: str


class AirQualityProvider(Protocol):
    def current(self, latitude: float, longitude: float) -> AirQualityReading: ...


@dataclass(frozen=True)
class FixtureAirQualityProvider:
    reading: AirQualityReading | None = None

    def current(self, latitude: float, longitude: float) -> AirQualityReading:
        if self.reading is None:
            raise RuntimeError("air-quality fixture is not configured")
        return AirQualityReading(
            latitude,
            longitude,
            self.reading.us_aqi,
            self.reading.pm2_5,
            self.reading.observed_at,
            self.reading.source,
        )


@dataclass(frozen=True)
class OpenMeteoAirQualityProvider:
    endpoint: str = "https://air-quality-api.open-meteo.com/v1/air-quality"
    timeout_seconds: float = 5.0

    def current(self, latitude: float, longitude: float) -> AirQualityReading:
        parsed = urllib.parse.urlsplit(self.endpoint)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("air-quality endpoint must use HTTPS")
        query = urllib.parse.urlencode(
            {
                "latitude": f"{latitude:.6f}",
                "longitude": f"{longitude:.6f}",
                "current": "us_aqi,pm2_5",
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
            raise RuntimeError("air-quality provider unavailable") from exc
        current = payload.get("current") if isinstance(payload, dict) else None
        if not isinstance(current, dict):
            raise RuntimeError("air-quality response has no current conditions")
        aqi, pm = current.get("us_aqi"), current.get("pm2_5")
        return AirQualityReading(
            latitude,
            longitude,
            float(aqi) if isinstance(aqi, (int, float)) else None,
            float(pm) if isinstance(pm, (int, float)) else None,
            str(current.get("time", ""))[:80],
            "open_meteo_air_quality",
        )


def configured_air_quality_provider() -> AirQualityProvider:
    raw = os.environ.get("AEGIS_AIR_QUALITY_FIXTURE_JSON")
    if raw:
        try:
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError
            return FixtureAirQualityProvider(
                AirQualityReading(
                    0,
                    0,
                    float(value["us_aqi"]) if value.get("us_aqi") is not None else None,
                    float(value["pm2_5"]) if value.get("pm2_5") is not None else None,
                    str(value.get("observed_at", "fixture")),
                    "fixture_air_quality",
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("air-quality fixture configuration is invalid") from exc
    return OpenMeteoAirQualityProvider(
        os.environ.get("AEGIS_AIR_QUALITY_ENDPOINT", OpenMeteoAirQualityProvider.endpoint)
    )


def air_quality_evidence(reading: AirQualityReading) -> dict[str, object]:
    return {
        "source": reading.source,
        "latitude": reading.latitude,
        "longitude": reading.longitude,
        "us_aqi": reading.us_aqi,
        "pm2_5": reading.pm2_5,
        "observed_at": reading.observed_at,
    }
