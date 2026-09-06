import json

import pytest

from aegis.weather import (
    FixtureWeatherProvider,
    OpenMeteoProvider,
    WeatherReading,
    weather_evidence,
    weather_forecast_evidence,
)


def test_fixture_weather_is_bounded_and_reprojects_coordinates() -> None:
    reading = FixtureWeatherProvider(
        WeatherReading(0, 0, 21.5, 20.0, 1, "2026-09-05T12:00", "fixture")
    ).current(41.88, -87.63)
    assert reading.latitude == 41.88
    assert reading.longitude == -87.63
    assert weather_evidence(reading)["temperature_c"] == 21.5


def test_open_meteo_provider_parses_current_conditions(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit):
            return json.dumps(
                {
                    "current": {
                        "time": "2026-09-05T12:00",
                        "temperature_2m": 22.0,
                        "apparent_temperature": 21.0,
                        "weather_code": 2,
                    }
                }
            ).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    result = OpenMeteoProvider().current(41.88, -87.63)
    assert result.temperature_c == 22.0
    assert result.weather_code == 2
    assert result.source == "open_meteo"


def test_open_meteo_rejects_non_https_endpoint() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        OpenMeteoProvider("http://weather.invalid").current(0, 0)


def test_open_meteo_provider_parses_bounded_daily_forecast(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit):
            return json.dumps(
                {
                    "daily": {
                        "time": ["2026-09-05", "2026-09-06"],
                        "temperature_2m_max": [24.0, 25.0],
                        "temperature_2m_min": [16.0, 17.0],
                        "precipitation_probability_max": [30, 45],
                        "sunrise": ["2026-09-05T11:20", "2026-09-06T11:19"],
                        "sunset": ["2026-09-06T00:20", "2026-09-07T00:21"],
                    }
                }
            ).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    result = OpenMeteoProvider().forecast(41.88, -87.63, 2)
    assert len(result) == 2
    assert result[0].temperature_max_c == 24.0
    assert result[1].precipitation_probability_max == 45.0
    assert weather_forecast_evidence(result)[0]["source"] == "open_meteo"


def test_open_meteo_forecast_rejects_invalid_horizon() -> None:
    with pytest.raises(ValueError, match="between 1 and 7"):
        OpenMeteoProvider().forecast(0, 0, 8)
