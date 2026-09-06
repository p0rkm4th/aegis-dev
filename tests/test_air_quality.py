import json

import pytest

from aegis.air_quality import OpenMeteoAirQualityProvider


def test_open_meteo_air_quality_parses_current_conditions(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit):
            return json.dumps(
                {
                    "current": {
                        "time": "2026-09-06T03:00",
                        "us_aqi": 38,
                        "pm2_5": 4.2,
                    }
                }
            ).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    result = OpenMeteoAirQualityProvider().current(41.88, -87.62)
    assert result.us_aqi == 38
    assert result.pm2_5 == 4.2
    assert result.source == "open_meteo_air_quality"


def test_open_meteo_air_quality_rejects_non_https_endpoint() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        OpenMeteoAirQualityProvider("http://air.invalid").current(0, 0)
