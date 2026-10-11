"""Load weather_raw.json, validate it, and print daily summaries."""

import json
from pathlib import Path

from weatherkit.records import to_readings
from weatherkit.schemas import WeatherResponse
from weatherkit.summarize import DailyAggregator


def main() -> None:
    """Load, validate, aggregate, and print daily weather summaries."""
    raw = json.loads(Path("weather_raw.json").read_text())
    response = WeatherResponse.model_validate(raw)
    readings = to_readings(response)

    agg = DailyAggregator()
    summaries = agg.summarize(readings)

    print(f"{'date':<12}{'high':>7}{'low':>7}{'precip':>9}{'range':>8}{'hours':>7}")
    for s in summaries:
        print(
            f"{s.date:<12}{s.temp_max:>7.1f}{s.temp_min:>7.1f}"
            f"{s.precipitation_sum:>9.1f}{s.temp_range():>8.1f}{s.hours_observed:>7}"
        )

    incomplete = agg.incomplete_days(readings)
    if incomplete:
        print(f"\nWarning: dropped incomplete days: {', '.join(incomplete)}")


if __name__ == "__main__":
    main()

# Q: What would happen if the guard were omitted and someone imported report.py to reuse a helper?
# A: The script would run on import -- printing the table and possibly crashing -- every time anyone imports report.py.

# --- Reflection ---
#
# Q1: WeatherResponse rejects the whole file if a single temperature is null.
#     Is that the right behavior?
# A: It depends. Rejecting is right for an automated pipeline where a silent gap
#    could corrupt downstream summaries. Tolerating is better for dashboards where
#    a missing hour is fine and you'd rather show the other 167. To tolerate it,
#    I'd change temperature_2m to list[float | None] and have to_readings skip Nones.
#
# Q2: DailyAggregator.min_hours defaults to 24. What if a pipeline runs at noon?
# A: The current day would have only ~12 hours and get dropped, so the day's max/min
#    would be based on partial data. incomplete_days() surfaces that date so nothing
#    disappears silently -- the caller can decide whether to keep the partial day.
#
# Q3: Name one concrete thing that becomes easier in Week 10 because weatherkit is a package.
# A: The cloud pipeline can `from weatherkit import DailyAggregator, to_readings` and
#    reuse the same validated logic, instead of copy-pasting code from a one-off script.