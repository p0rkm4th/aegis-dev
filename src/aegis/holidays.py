"""Bounded public-holiday reads for planning; never canonical personal events."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PublicHoliday:
    date: str
    name: str
    country_code: str
    global_holiday: bool
    types: tuple[str, ...]
    source: str


class HolidayProvider(Protocol):
    def list_holidays(self, country_code: str, year: int) -> tuple[PublicHoliday, ...]: ...


@dataclass(frozen=True)
class FixtureHolidayProvider:
    holidays: tuple[PublicHoliday, ...]

    def list_holidays(self, country_code: str, year: int) -> tuple[PublicHoliday, ...]:
        del year
        return tuple(
            PublicHoliday(
                item.date, item.name, country_code, item.global_holiday, item.types, item.source
            )
            for item in self.holidays[:50]
        )


@dataclass(frozen=True)
class NagerHolidayProvider:
    endpoint: str = "https://date.nager.at/api/v3/PublicHolidays"
    timeout_seconds: float = 5.0

    def list_holidays(self, country_code: str, year: int) -> tuple[PublicHoliday, ...]:
        parsed = urllib.parse.urlsplit(self.endpoint)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("holiday endpoint must use HTTPS")
        url = f"{self.endpoint.rstrip('/')}/{year}/{urllib.parse.quote(country_code, safe='')}"
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                values = json.loads(response.read(400_000))
        except Exception as exc:
            raise RuntimeError("public holiday provider unavailable") from exc
        if not isinstance(values, list):
            raise RuntimeError("public holiday response is invalid")
        result: list[PublicHoliday] = []
        for value in values[:50]:
            if not isinstance(value, dict):
                continue
            date, name = value.get("date"), value.get("name")
            if not isinstance(date, str) or not isinstance(name, str):
                continue
            types = value.get("types")
            result.append(
                PublicHoliday(
                    date[:20],
                    name[:300],
                    country_code,
                    bool(value.get("global", False)),
                    tuple(str(item)[:40] for item in types[:10]) if isinstance(types, list) else (),
                    "nager_date",
                )
            )
        return tuple(result)


def configured_holiday_provider() -> HolidayProvider:
    raw = os.environ.get("AEGIS_HOLIDAY_FIXTURE_JSON")
    if raw:
        try:
            values = json.loads(raw)
            if not isinstance(values, list):
                raise ValueError
            holidays = tuple(
                PublicHoliday(
                    str(item["date"]),
                    str(item["name"]),
                    "fixture",
                    bool(item.get("global", True)),
                    tuple(str(kind) for kind in item.get("types", [])),
                    "fixture_holidays",
                )
                for item in values[:50]
                if isinstance(item, dict)
            )
            return FixtureHolidayProvider(holidays)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("holiday fixture configuration is invalid") from exc
    return NagerHolidayProvider(
        os.environ.get("AEGIS_HOLIDAY_ENDPOINT", NagerHolidayProvider.endpoint)
    )


def holidays_evidence(holidays: tuple[PublicHoliday, ...]) -> dict[str, object]:
    return {
        "source": holidays[0].source if holidays else "public_holiday_provider",
        "holidays": [
            {
                "date": item.date,
                "name": item.name,
                "country_code": item.country_code,
                "global": item.global_holiday,
                "types": list(item.types),
            }
            for item in holidays[:50]
        ],
    }
