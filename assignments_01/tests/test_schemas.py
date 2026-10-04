"""Tests for weatherkit.schemas."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from weatherkit.schemas import WeatherResponse

# Path(__file__).parent.parent / "weather_raw.json" works no matter which
# directory pytest is run from; a plain relative path would break outside assignments_01/.
DATA_PATH = Path(__file__).parent.parent / "weather_raw.json"


def test_valid_response_loads():
    raw = json.loads(DATA_PATH.read_text())
    response = WeatherResponse.model_validate(raw)
    assert len(response.hourly.time) == 168


def test_latitude_out_of_range():
    raw = json.loads(DATA_PATH.read_text())
    raw["latitude"] = 200.0
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(raw)


def test_mismatched_list_lengths():
    bad = {
        "latitude": 35.0,
        "longitude": -80.0,
        "timezone": "GMT",
        "elevation": 200.0,
        "hourly": {
            "time": ["2026-04-08T00:00", "2026-04-08T01:00", "2026-04-08T02:00"],
            "temperature_2m": [10.0, 11.0],                # one short on purpose
            "precipitation": [0.0, 0.0, 0.0],
        },
    }
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(bad)


def test_null_temperature_raises():
    bad = {
        "latitude": 35.0,
        "longitude": -80.0,
        "timezone": "GMT",
        "elevation": 200.0,
        "hourly": {
            "time": ["2026-04-08T00:00"],
            "temperature_2m": [None],
            "precipitation": [0.0],
        },
    }
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(bad)