"""The weatherkit package: turn raw weather API responses into daily summaries."""

from weatherkit.records import HourlyReading, to_readings
from weatherkit.schemas import HourlyBlock, WeatherResponse
from weatherkit.summarize import DailyAggregator, DailySummary

__all__ = [
    "HourlyReading",
    "to_readings",
    "HourlyBlock",
    "WeatherResponse",
    "DailyAggregator",
    "DailySummary",
]