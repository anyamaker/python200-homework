"""Group hourly readings into daily summaries."""

from collections import defaultdict
from dataclasses import dataclass

from weatherkit.records import HourlyReading


@dataclass(frozen=True)
class DailySummary:
    """Aggregated stats for one calendar day.

    date: YYYY-MM-DD string
    temp_max / temp_min: Celsius
    precipitation_sum: millimeters
    hours_observed: how many hourly readings contributed
    """
    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        """Difference between the day's max and min temperature."""
        return self.temp_max - self.temp_min


class DailyAggregator:
    """Group hourly readings into daily summaries.

    Days with fewer than min_hours readings are dropped.
    """

    def __init__(self, min_hours: int = 24) -> None:
        self.min_hours = min_hours

    def _group(self, readings: list[HourlyReading]) -> dict[str, list[HourlyReading]]:
        """Group readings by the date part of their timestamp."""
        groups: dict[str, list[HourlyReading]] = defaultdict(list)
        for r in readings:
            groups[r.timestamp[:10]].append(r)
        return groups

    def summarize(self, readings: list[HourlyReading]) -> list[DailySummary]:
        """Return one DailySummary per complete day, sorted by date.

        Args:
            readings: A list of HourlyReading objects.

        Returns:
            DailySummary objects for days with at least min_hours readings,
            sorted by date. Incomplete days are omitted.
        """
        summaries: list[DailySummary] = []
        for date, day_readings in self._group(readings).items():
            if len(day_readings) < self.min_hours:
                continue
            temps = [r.temperature_c for r in day_readings]
            precip = sum(r.precipitation_mm for r in day_readings)
            summaries.append(
                DailySummary(
                    date=date,
                    temp_max=max(temps),
                    temp_min=min(temps),
                    precipitation_sum=precip,
                    hours_observed=len(day_readings),
                )
            )
        return sorted(summaries, key=lambda s: s.date)

    def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        """Return dates that were dropped because they had too few hours.

        Args:
            readings: A list of HourlyReading objects.

        Returns:
            Sorted list of date strings that didn't meet min_hours.
        """
        dropped = [
            date for date, day_readings in self._group(readings).items()
            if len(day_readings) < self.min_hours
        ]
        return sorted(dropped)
