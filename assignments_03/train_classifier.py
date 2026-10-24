"""Week 3 mini-project: train a weather classifier and save it."""

import requests
import pandas as pd
import json
import sklearn
import joblib

# --- Task 1: Fetch the Data ---
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 36.17,
    "longitude": -115.14,
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/Los_Angeles",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print("Shape:", df.shape)
print(df.head())

# City: Las Vegas, NV

# --- Task 2: Engineer the Label ---

def is_good_for_running(row):
    """Return 1 if the day is good for running, 0 otherwise."""
    return int(
        (7 <= row["temperature_2m_max"] <= 26)
        and (row["temperature_2m_min"] >= 0)
        and (row["precipitation_sum"] < 3)
        and (row["wind_speed_10m_max"] < 30)
    )

df["good_for_running"] = df.apply(is_good_for_running, axis=1)

good = df["good_for_running"].sum()
skip = len(df) - good
print(f"Good days: {good}  ({good/len(df)*100:.1f}%)")
print(f"Skip days: {skip}  ({skip/len(df)*100:.1f}%)")

# Q: What fraction of days are good, and does that seem reasonable for Las Vegas?
# A: 38% good, 62% skip. Las Vegas is hot and dry: summer highs above 26 C push many days to skip,
#    plus occasional wind and the rare rain. The split is moderately imbalanced (roughly 40/60),
#    so accuracy alone is not enough -- precision and recall matter too, because a lazy model
#    that always says "skip" would already score 62%.

# --- Task 3: Build and Evaluate a KNN Classifier ---

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

feature_cols = ["temperature_2m_max", "temperature_2m_min",
                "precipitation_sum", "wind_speed_10m_max"]
X = df[feature_cols]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

k_values = [1, 3, 5, 7, 9, 11, 15, 21]
best_k = None
best_score = -1.0
for k in k_values:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    mean_score = scores.mean()
    print(f"k={k:2d}  mean CV accuracy = {mean_score:.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print(f"\nChosen k = {best_k}  (CV accuracy {best_score:.4f})")

knn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k)),
])
knn_pipe.fit(X_train, y_train)
knn_pred = knn_pipe.predict(X_test)
print("KNN test accuracy:", accuracy_score(y_test, knn_pred))
print("KNN classification report:")
print(classification_report(y_test, knn_pred))

# Q: Why did k=1 win here, and is that safe?
# A: CV picked k=1 (0.9384), with scores decreasing monotonically as k grows.
#    The "good" region is a rectangular box in feature space, and k=1 traces its corners exactly.
#    Larger k smooths those corners away. The margin over k=3 is tiny (0.36%), so either is
#    defensible, but k=1 also scored 0.945 on the held-out test set -- so no real overfitting here.

# --- Task 4: Build and Evaluate a Logistic Regression Classifier ---

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

log_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])
log_pipe.fit(X_train, y_train)

log_pred = log_pipe.predict(X_test)
log_proba = log_pipe.predict_proba(X_test)[:, 1]

print("\nLogReg test accuracy:", accuracy_score(y_test, log_pred))
print("LogReg precision:", precision_score(y_test, log_pred))
print("LogReg recall:", recall_score(y_test, log_pred))
print("LogReg F1:", f1_score(y_test, log_pred))
print("LogReg classification report:")
print(classification_report(y_test, log_pred))

print("\nLogReg coefficients:")
logreg = log_pipe.named_steps["logreg"]
for name, coef in zip(feature_cols, logreg.coef_[0]):
    print(f"  {name:22s}: {coef:+.3f}")

# Q: Which conditions most strongly push a day toward "skip"?
# A: temperature_2m_max has the strongest negative coefficient (-2.60). In Las Vegas,
#    high temp_max means summer heat above 26 C, which is the main skip condition.
#    Precipitation is second (-0.79). Wind is almost neutral because it rarely exceeds 30 km/h here.

# --- Task 5: Compare and Choose ---
#
# Q: Which model had higher accuracy?
# A: KNN (0.945) vs LogReg (0.932). KNN also won on recall and F1; LogReg only on precision.
#
# Q: For a running app, is accuracy the metric I care about most?
# A: No -- I'd weigh recall more. Missing a good day (false negative) hurts user trust more
#    than sending them out in the rain once.
#
# Q: Which model will I deploy, and why?
# A: KNN. Higher accuracy, recall, and F1. Trade-offs I accept: bigger file, slower prediction,
#    and coarser probabilities. If I needed interpretability and tiny files, I'd pick LogReg.

# --- Task 6: Save the Model ---

final_model = knn_pipe
final_accuracy = accuracy_score(y_test, knn_pipe.predict(X_test))
final_f1 = f1_score(y_test, knn_pipe.predict(X_test))

joblib.dump(final_model, "models/weather_classifier.pkl")

metadata = {
    "python_version": ".".join(map(str, __import__("sys").version_info[:3])),
    "sklearn_version": sklearn.__version__,
    "features": feature_cols,
    "label_rule": "good if 7 <= temp_max <= 26, temp_min >= 0, precip < 3 mm, wind < 30 km/h",
    "city": "Las Vegas, NV",
    "test_accuracy": float(final_accuracy),
    "test_f1": float(final_f1),
    "model_type": "KNN (k=1) + StandardScaler",
}

with open("models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved model to models/weather_classifier.pkl")
print("Saved metadata to models/weather_classifier_metadata.json")
