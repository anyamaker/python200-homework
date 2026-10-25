"""Script that uses WeatherClassifier to predict running weather."""

from weather_model import WeatherClassifier


def main():
    """Load the saved model and predict for a few hypothetical days."""
    clf = WeatherClassifier("models/weather_classifier.pkl")

    days = [
        # clearly good: mild, dry, calm
        {"temperature_2m_max": 22.0, "temperature_2m_min": 12.0,
         "precipitation_sum": 0.0, "wind_speed_10m_max": 5.0},
        # clearly bad: cold, rainy, windy
        {"temperature_2m_max": 5.0, "temperature_2m_min": -2.0,
         "precipitation_sum": 10.0, "wind_speed_10m_max": 40.0},
        # borderline hot: 27 C, right past the 26 C cutoff
        {"temperature_2m_max": 27.0, "temperature_2m_min": 15.0,
         "precipitation_sum": 0.0, "wind_speed_10m_max": 8.0},
        # borderline windy: 29 km/h, just under the 30 cutoff
        {"temperature_2m_max": 21.0, "temperature_2m_min": 11.0,
         "precipitation_sum": 0.0, "wind_speed_10m_max": 29.0},
    ]

    predictions = clf.predict(days)

    print("Predictions:")
    for day, p in zip(days, predictions):
        print(
            f"  max={day['temperature_2m_max']:>4}  "
            f"min={day['temperature_2m_min']:>4}  "
            f"precip={day['precipitation_sum']:>4}  "
            f"wind={day['wind_speed_10m_max']:>4}  ->  "
            f"{p.label:>4}  (P(good) = {p.probability:.2f})"
        )


if __name__ == "__main__":
    main()

# Q: What would happen if the guard were omitted and another script imported predict_weather.py?
# A: main() would run on import -- the predictions would print every time anyone imports the module,
#    even if they only wanted to reuse a helper function.


# --- Task 6: Reflection ---
#
# Q: predict() selects the four features in a fixed order. Describe a concrete bug this prevents.
# A: If a caller passed the dict in a different column order (e.g. precipitation first),
#    the model would interpret precipitation values as temperature values and produce wrong labels
#    with no error at all. Fixing the order in FEATURES prevents that silent bug.
#
# Q: The graded tests train a tiny model in a fixture instead of loading models/weather_classifier.pkl.
#    Why is that a better design?
# A: The tests stay self-contained: they do not depend on a specific file being on disk, they run
#    fast, and they still pass on a fresh clone where models/ might be missing or retrained.
#
# Q: In Week 10 a pipeline will import WeatherClassifier and call predict() on batches from a database.
#    Name one thing that makes it ready, and one thing you might add before production.
# A: Ready: predict() already accepts a list of dicts and returns a list of Prediction objects,
#    which is exactly the batch shape a pipeline needs.
#    Add before production: batch input validation (e.g. reject NaN or wrong types) and better
#    handling of extreme days, where KNN with k=1 is unreliable.