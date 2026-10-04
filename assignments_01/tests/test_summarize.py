"""Tests for weatherkit.summarize."""

import pytest

from weatherkit.records import HourlyReading
from weatherkit.summarize import DailyAggregator


@pytest.fixture
def two_day_readings() -> list[HourlyReading]:
    """Two full days of hourly readings (24 hours each)."""
    day1 = [
        HourlyReading(timestamp=f"2026-04-08T{h:02d}:00", temperature_c=float(h), precipitation_mm=0.5)
        for h in range(24)
    ]
    day2 = [
        HourlyReading(timestamp=f"2026-04-09T{h:02d}:00", temperature_c=float(h) + 10, precipitation_mm=1.0)
        for h in range(24)
    ]
    return day1 + day2


def test_grouping_by_date(two_day_readings):
    summaries = DailyAggregator().summarize(two_day_readings)
    assert len(summaries) == 2
    assert summaries[0].date == "2026-04-08"
    assert summaries[1].date == "2026-04-09"


def test_temp_max_min(two_day_readings):
    summaries = DailyAggregator().summarize(two_day_readings)
    assert summaries[0].temp_max == 23.0
    assert summaries[0].temp_min == 0.0
    assert summaries[1].temp_max == 33.0
    assert summaries[1].temp_min == 10.0


def test_precipitation_sum(two_day_readings):
    summaries = DailyAggregator().summarize(two_day_readings)
    assert summaries[0].precipitation_sum == pytest.approx(24 * 0.5)
    assert summaries[1].precipitation_sum == pytest.approx(24 * 1.0)


def test_incomplete_day_dropped_and_reported():
    short_day = [
        HourlyReading(timestamp=f"2026-04-08T{h:02d}:00", temperature_c=10.0, precipitation_mm=0.0)
        for h in range(6)  # only 6 hours
    ]
    agg = DailyAggregator(min_hours=24)
    assert agg.summarize(short_day) == []
    assert agg.incomplete_days(short_day) == ["2026-04-08"]


def test_lower_min_hours_keeps_day():
    short_day = [
        HourlyReading(timestamp=f"2026-04-08T{h:02d}:00", temperature_c=10.0, precipitation_mm=0.0)
        for h in range(6)
    ]
    agg = DailyAggregator(min_hours=6)
    summaries = agg.summarize(short_day)
    assert len(summaries) == 1
    assert summaries[0].hours_observed == 6


@pytest.mark.parametrize(
    "hours, expected_kept",
    [
        (23, False),
        (24, True),
        (25, True),
    ],
)
def test_min_hours_boundary(hours, expected_kept):
    readings = [
        HourlyReading(timestamp=f"2026-04-08T{h:02d}:00", temperature_c=10.0, precipitation_mm=0.0)
        for h in range(hours)
    ]
    summaries = DailyAggregator(min_hours=24).summarize(readings)
    assert (len(summaries) == 1) is expected_kept

# Breaking incomplete_days() from < to > made test_incomplete_day_dropped_and_reported fail.