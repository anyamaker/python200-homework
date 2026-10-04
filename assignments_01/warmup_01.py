import pytest
from dataclasses import dataclass, field, FrozenInstanceError
from pydantic import BaseModel, Field, ValidationError, model_validator


# --- Classes ---
# Q1

class Thermometer:
    def __init__(self, location, readings=None):
        self.location = location
        self.readings = readings if readings is not None else []

    def add(self, reading):
        self.readings.append(reading)

    def average(self):
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)

    def hottest(self):
        if not self.readings:
            return None
        return max(self.readings)

    def __repr__(self):
        avg = self.average()
        avg_str = f"{avg:.1f}" if avg is not None else "None"
        return f"Thermometer(location='{self.location}', n_readings={len(self.readings)}, average={avg_str})"


my_thermometer = Thermometer("Charlotte")
my_thermometer.add(15.5)
my_thermometer.add(18.2)
my_thermometer.add(22.0)
my_thermometer.add(19.4)

print(my_thermometer.average())
print(my_thermometer.hottest())

# Q: Why does average() need to handle the empty case?
# A: Without the check, sum([]) / 0 raises ZeroDivisionError and crashes.

# --- Q2 checks ---
print(my_thermometer)
t2 = Thermometer("Raleigh", [10.0, 12.5, 14.0])
print([my_thermometer, t2])

# Q: What does Python display when a class has no __repr__, and why is it unhelpful when debugging?
# A: It shows something like <__main__.Thermometer object at 0x104f3a1d0>, which tells you nothing about the object's data.

# --- Q3 ---

class TemperatureAlert:
    def __init__(self, threshold=30.0):
        self.threshold = threshold

    def breaches(self, thermometer):
        return [r for r in thermometer.readings if r > self.threshold]


# same thermometer, two different alerts
alert_high = TemperatureAlert()
alert_low = TemperatureAlert(25.0)

print(alert_high.breaches(my_thermometer))
print(alert_low.breaches(my_thermometer))

# Q: Why is the threshold stored on TemperatureAlert rather than passed as an argument to breaches()?
# A: So the same alert object can be reused across many thermometers without repeating the threshold everywhere; you just create different alert objects for different thresholds.


# --- Dataclasses ---
# Q1

@dataclass(frozen=True)
class Station:
    """A weather station with its location and elevation."""
    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float


station_a = Station("KCLT", "Charlotte Douglas", 35.21, -80.94, 234.0)
station_b = Station("KCLT", "Charlotte Douglas", 35.21, -80.94, 234.0)

print(station_a == station_b)

# Q: Why is the result what it is, and what would it have been with the original hand-written class?
# A: Dataclass compares by field values, so station_a == station_b is True. The original class would compare by identity, so it would be False.

# --- Q2 ---

try:
    station_a.elevation = 999.0
except FrozenInstanceError as e:
    print("FrozenInstanceError:", e)

stations = {station_a, station_b, Station("KRDU", "Raleigh-Durham", 35.88, -78.78, 133.0)}
print(len(stations))

# Q: What does frozen=True give you besides immutability, and why is that useful here?
# A: It makes instances hashable, so they can go in a set or dict; without it, two identical stations would be counted twice.

# --- Q3 ---

@dataclass
class StationBatch:
    """A group of stations in the same region."""
    region: str
    stations: list[Station] = field(default_factory=list)

    def add(self, station: Station) -> None:
        """Append a station to the batch."""
        self.stations.append(station)

    def highest(self) -> Station | None:
        """Return the station with the greatest elevation, or None if empty."""
        if not self.stations:
            return None
        return max(self.stations, key=lambda s: s.elevation)


batch = StationBatch("Southeast")
batch.add(Station("KCLT", "Charlotte Douglas", 35.21, -80.94, 234.0))
batch.add(Station("KRDU", "Raleigh-Durham", 35.88, -78.78, 133.0))
batch.add(Station("KAVL", "Asheville", 35.43, -82.54, 670.0))

print(batch.highest())
print(batch)

# Q: What error do you get with stations: list[Station] = []?
# A: ValueError: mutable default <class 'list'> for field stations is not allowed: use default_factory

# Q: Why does Python refuse stations: list[Station] = []?
# A: A mutable default is shared across every instance, so all StationBatch objects would append to the same list; default_factory creates a fresh list per object.


# --- Pydantic ---
# Q1

class Reading(BaseModel):
    """A single weather reading from a station."""
    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def check_sensor(self):
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError("humidity 0 at temp < -40 looks like a broken sensor")
        return self


reading = Reading(
    station_id="KCLT",
    timestamp="2026-04-08T12:00",
    temperature_c=21.5,
    humidity=65.0,
)
print(reading)

# --- Q2 ---

# missing required field (no timestamp)
try:
    Reading(station_id="KCLT", temperature_c=21.5, humidity=65.0)
except ValidationError as e:
    print("Missing field:", e)

# temperature out of range
try:
    Reading(station_id="KCLT", timestamp="2026-04-08T12:00", temperature_c=150.0, humidity=65.0)
except ValidationError as e:
    print("Bad temperature:", e)

# humidity not a number
try:
    Reading(station_id="KCLT", timestamp="2026-04-08T12:00", temperature_c=21.5, humidity="very humid")
except ValidationError as e:
    print("Bad humidity:", e)

# coercion demo
coerced = Reading(
    station_id="KCLT",
    timestamp="2026-04-08T12:00",
    temperature_c="21.5",
    humidity=40,
)
print(coerced)
print(type(coerced.temperature_c), type(coerced.humidity))

# Q: Why does Pydantic accept "21.5" but reject "very humid"?
# A: Pydantic tries to coerce a string to float, and "21.5" is a valid number, while "very humid" is not.

# --- Q3 ---

try:
    Reading(
        station_id="KL",              # too short
        # timestamp missing
        temperature_c="not a number", # not numeric
        humidity=50.0,
    )
except ValidationError as e:
    for err in e.errors():
        print(err["loc"], "-", err["msg"])

# Q: How many errors were reported, and why is reporting all of them at once more useful than stopping at the first?
# A: Three errors (station_id, timestamp, temperature_c); reporting all at once lets you fix everything in one pass instead of rerunning for each bug.

# --- Q4 ---

# valid reading still constructs
ok = Reading(
    station_id="KCLT",
    timestamp="2026-04-08T12:00",
    temperature_c=-50.0,
    humidity=20.0,
)
print("Valid:", ok)

# bad combination raises
try:
    Reading(
        station_id="KCLT",
        timestamp="2026-04-08T12:00",
        temperature_c=-50.0,
        humidity=0.0,
    )
except ValidationError as e:
    print("Bad sensor:", e)

# Q: Why can't this rule be expressed with Field constraints alone?
# A: Field constraints only check one field at a time; this rule depends on the combination of humidity and temperature_c, so it needs a model_validator.


# --- pytest ---

def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32


def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)

# Q: Why was pytest.approx necessary for the 37 C case?
# A: Floating-point math gives 98.60000000000001, so plain == fails; approx compares within a tiny tolerance.


def mean(values: list[float]) -> float:
    """Return the mean of a non-empty list of numbers."""
    if not values:
        raise ValueError("mean() requires a non-empty list")
    return sum(values) / len(values)


def test_mean_of_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        mean([])

# Q: What would pytest.raises(ValueError) alone fail to catch that match= catches?
# A: It would pass for any ValueError, even if the message says something unrelated; match= checks the message text too.


@pytest.mark.parametrize(
    "values, expected",
    [
        ([10.0], 10.0),                    # single element
        ([1.0, 2.0, 3.0], 2.0),            # simple average
        ([-5.0, 5.0], 0.0),                # negative + positive
        ([-10.0, -20.0], -15.0),           # all negative
    ],
)
def test_mean_values(values, expected):
    assert mean(values) == pytest.approx(expected)

# Q: Why is one parametrized test with four cases better than four nearly identical test functions?
# A: Less duplication, one place to update, and each case still shows up separately in the pytest output.

# pytest summary: 6 passed in 0.04s

# Broken output: assert 257.0 == 212 (celsius_to_fahrenheit(100) returned 257.0 with 9/4 instead of 9/5)

# Q: What specific values did pytest show in the failure report, and why is that more useful than a bare "assertion failed"?
# A: It showed both sides (257.0 vs 212) and which call produced them, so you can see exactly where the math went wrong instead of just "it failed".
