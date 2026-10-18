"""Week 3 mini-project: load the saved model and predict good_for_running."""

import json
import joblib
import pandas as pd

# --- Task 7: Load and Predict ---

model = joblib.load("models/weather_classifier.pkl")
with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)

print("Loaded model metadata:")
print("  city:          ", metadata["city"])
print("  features:      ", metadata["features"])
print("  test accuracy: ", metadata["test_accuracy"])
print("  test F1:       ", metadata["test_f1"])
print("  model type:    ", metadata["model_type"])

# Four hypothetical days: clearly good, clearly bad, and two borderline ones.
days = pd.DataFrame([
    # good: mild, dry, calm
    {"temperature_2m_max": 20.0, "temperature_2m_min": 10.0,
     "precipitation_sum": 0.0, "wind_speed_10m_max": 10.0},
    # bad: rainy, windy, cold
    {"temperature_2m_max": 5.0, "temperature_2m_min": -2.0,
     "precipitation_sum": 8.0, "wind_speed_10m_max": 40.0},
    # borderline: hot but dry (near the 26 C upper bound)
    {"temperature_2m_max": 27.0, "temperature_2m_min": 15.0,
     "precipitation_sum": 0.0, "wind_speed_10m_max": 12.0},
    # borderline: pleasant but slightly windy (near the 30 km/h bound)
    {"temperature_2m_max": 22.0, "temperature_2m_min": 12.0,
     "precipitation_sum": 0.5, "wind_speed_10m_max": 29.0},
])

# Reorder columns to match the training order
days = days[metadata["features"]]

predictions = model.predict(days)
probabilities = model.predict_proba(days)[:, 1]

print("\nPredictions:")
for i, (idx, row) in enumerate(days.iterrows()):
    label = "good" if predictions[i] == 1 else "skip"
    print(f"  day {i+1}: {row.to_dict()}  ->  {label}  (P(good) = {probabilities[i]:.2f})")

# --- Task 8: Reflect ---
#
# Q: Pick your borderline day. What probability did the model give it? Confident or uncertain?
#    How would you handle a day where it predicts 0.52?
# A: Day 4 (pleasant but wind 29) got P(good)=0.00 -- the model was "confident" it was skip,
#    but it was actually wrong by the rule. With k=1, probabilities are always 0 or 1,
#    so there is no real confidence signal. A 0.52 from this model is impossible, but if I
#    had a model that produced 0.52, I would treat it as "uncertain" and either show a warning
#    to the user ("might be OK, check the sky") or fall back to a simpler rule.
#
# Q: predict.py has no training code. What breaks if someone runs it before train_classifier.py?
#    How could the error message be more helpful?
# A: joblib.load would raise FileNotFoundError because models/weather_classifier.pkl does not exist.
#    The message would be "No such file or directory". I could wrap the load in a try/except
#    and print "Run train_classifier.py first to create the model file" instead.
#
# Q: In Week 10 this model runs in a cloud pipeline. Name one thing that would need to change.
# A: The hard-coded file path "models/weather_classifier.pkl" would need to become a configurable
#    path or a cloud storage URI (e.g. S3), so the pipeline can load the model from wherever it lives.

