import json

import pytest

from aegis.weather import (
    FixtureWeatherProvider,
    OpenMeteoProvider,
    WeatherReading,
    weather_evidence,
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
