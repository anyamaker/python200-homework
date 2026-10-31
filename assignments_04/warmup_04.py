import json

# --- Warmup Q1 ---

class ModelInfo:
    """Loads a model metadata JSON file once and exposes basic info."""

    def __init__(self, path: str):
        with open(path) as f:
            self.data = json.load(f)

    def feature_count(self) -> int:
        """Return how many features the model expects."""
        return len(self.data["features"])


# Create a small JSON file for testing
test_meta = {
    "features": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"],
    "city": "Las Vegas, NV",
}
test_path = "warmup_meta.json"
with open(test_path, "w") as f:
    json.dump(test_meta, f)

# Build ModelInfo and print the feature count
info = ModelInfo(test_path)
print("Feature count:", info.feature_count())

# Q: Why load the file in __init__ instead of inside feature_count() if it is called many times?
# A: Because reading a file from disk is slow. Loading once in __init__ means every feature_count()
#    call is just a dict lookup, not a disk read. If it were inside the method, every call would
#    re-open the file.

# --- Warmup Q2 ---

from dataclasses import dataclass


@dataclass
class DayForecast:
    """A single day's forecast."""
    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        """Return a readable one-line summary."""
        rain_text = "rain expected" if self.will_rain else "no rain"
        return f"{self.date}: high {self.high_c}C, {rain_text}"


day1 = DayForecast(date="2025-06-15", high_c=24.1, will_rain=False)
day2 = DayForecast(date="2025-06-16", high_c=18.5, will_rain=True)

print(day1.summary())
print(day2.summary())

# Q: Name one thing @dataclass wrote for you that you did not have to write by hand.
# A: __init__ (so I can call DayForecast(date=..., high_c=..., will_rain=...) without writing it),
#    plus __repr__ and __eq__. I only write the field names and types.

# --- Warmup Q3 ---

import pytest


def parse_temperature(text: str) -> float:
    """Convert a string like '24.1' to a float.

    Raises:
        ValueError: if the text is not a valid number.
    """
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"Not a valid temperature: {text!r}")


def test_parse_temperature_valid():
    assert parse_temperature("24.1") == pytest.approx(24.1)


def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")

# Q: What would pytest.raises(ValueError) without match fail to catch?
# A: It would pass for any ValueError, even one caused by an unrelated bug.
#    match="temperature" makes sure the specific error message we expect is raised.
