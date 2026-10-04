"""Pydantic models describing the Open-Meteo API response."""

from pydantic import BaseModel, Field, model_validator


class HourlyBlock(BaseModel):
    """Columnar hourly arrays from the Open-Meteo response.

    time[i], temperature_2m[i] and precipitation[i] all describe the same hour.
    """
    time: list[str]
    temperature_2m: list[float]
    precipitation: list[float]

    @model_validator(mode="after")
    def check_lengths(self):
        if not (len(self.time) == len(self.temperature_2m) == len(self.precipitation)):
            raise ValueError("time, temperature_2m and precipitation must have the same length")
        return self


class WeatherResponse(BaseModel):
    """Top-level Open-Meteo response with metadata and the hourly block."""
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
    elevation: float
    hourly: HourlyBlock