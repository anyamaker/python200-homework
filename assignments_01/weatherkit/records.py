"""Internal data structures for weather data (inside the boundary)."""

from dataclasses import dataclass

from weatherkit.schemas import WeatherResponse


@dataclass(frozen=True)
class HourlyReading:
    """A single hourly observation.

    timestamp: ISO 8601 string, e.g. "2026-04-08T00:00"
    temperature_c: air temperature at 2m, in Celsius
    precipitation_mm: precipitation in millimeters
    """
    timestamp: str
    temperature_c: float
    precipitation_mm: float


def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert a columnar hourly block into a list of HourlyReading objects.

    Args:
        response: A validated WeatherResponse.

    Returns:
        One HourlyReading per hour, in the same order as the input arrays.
    """
    hourly = response.hourly
    return [
        HourlyReading(
            timestamp=hourly.time[i],
            temperature_c=hourly.temperature_2m[i],
            precipitation_mm=hourly.precipitation[i],
        )
        for i in range(len(hourly.time))
    ]

# Q: Why is HourlyReading a dataclass rather than a Pydantic model?
# A: It's built from already-validated data inside the program, so it only needs to hold values -- no validation required.
