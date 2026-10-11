"""Tests for weatherkit.records."""

import json
from pathlib import Path

from weatherkit.records import HourlyReading, to_readings
from weatherkit.schemas import WeatherResponse

DATA_PATH = Path(__file__).parent.parent / "weather_raw.json"


def _load_response() -> WeatherResponse:
    raw = json.loads(DATA_PATH.read_text())
    return WeatherResponse.model_validate(raw)


def test_to_readings_one_per_hour_in_order():
    response = _load_response()
    readings = to_readings(response)
    assert len(readings) == len(response.hourly.time)
    assert readings[0].timestamp == response.hourly.time[0]
    assert readings[-1].timestamp == response.hourly.time[-1]


def test_values_match_index():
    response = _load_response()
    readings = to_readings(response)
    for i in (0, 5, 100, 167):
        assert readings[i].timestamp == response.hourly.time[i]
        assert readings[i].temperature_c == response.hourly.temperature_2m[i]
        assert readings[i].precipitation_mm == response.hourly.precipitation[i]


def test_identical_readings_compare_equal():
    a = HourlyReading(timestamp="2026-04-08T00:00", temperature_c=16.8, precipitation_mm=0.0)
    b = HourlyReading(timestamp="2026-04-08T00:00", temperature_c=16.8, precipitation_mm=0.0)
    assert a == b