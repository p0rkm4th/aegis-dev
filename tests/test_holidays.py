import json

import pytest

from aegis.holidays import FixtureHolidayProvider, NagerHolidayProvider, PublicHoliday


def test_fixture_holidays_reprojects_country_and_bounds() -> None:
    result = FixtureHolidayProvider(
        (PublicHoliday("2026-12-25", "Christmas Day", "fixture", True, ("Public",), "fixture"),)
    ).list_holidays("US", 2026)
    assert result[0].country_code == "US"
    assert result[0].name == "Christmas Day"


def test_nager_provider_parses_public_holidays(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit):
            return json.dumps(
                [
                    {
                        "date": "2026-12-25",
                        "name": "Christmas Day",
                        "global": True,
                        "types": ["Public"],
                    }
                ]
            ).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    result = NagerHolidayProvider().list_holidays("US", 2026)
    assert result[0].country_code == "US"
    assert result[0].source == "nager_date"


def test_nager_provider_rejects_non_https_endpoint() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        NagerHolidayProvider("http://holidays.invalid").list_holidays("US", 2026)
