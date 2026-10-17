"""Week 2 mini-project: predict daily high temperature from other weather features."""

import requests
import pandas as pd

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
print("Columns:", list(df.columns))
print("First 5 rows:")
print(df.head())

# City: Las Vegas, NV


# --- Task 2: Look at the Features ---

print("\n--- describe ---")
print(df.describe())

print("\n--- missing values ---")
print(df.isna().sum())

print("\n--- correlation with temperature_2m_max ---")
correlations = df.corr(numeric_only=True)["temperature_2m_max"].sort_values(ascending=False)
print(correlations)


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Task 2: plot ---
plt.figure(figsize=(7, 5))
plt.scatter(df["temperature_2m_min"], df["temperature_2m_max"], alpha=0.5)
plt.title("Daily High vs Low")
plt.xlabel("temperature_2m_min (C)")
plt.ylabel("temperature_2m_max (C)")
plt.savefig("outputs/high_vs_low.png", dpi=100, bbox_inches="tight")
plt.close()

# Q: What do you see?
# A: One long cloud of points going up diagonally. Warm nights -> warm days.
#    No second band like in the rain example, just one straight trend.

# --- Task 3: Engineer a Feature ---

df["is_summer"] = df["date"].dt.month.isin([6, 7, 8]).astype(int)

summer_count = df["is_summer"].sum()
non_summer_count = len(df) - summer_count
print(f"Summer days: {summer_count}")
print(f"Non-summer days: {non_summer_count}")

# Q: Do you expect is_summer to help predict the daily high, and why?
# A: Yes. Summer days are much hotter, so this flag should push the prediction up.

# --- Task 4: Baseline Model ---

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

X_base = df[["temperature_2m_min"]].values
y = df["temperature_2m_max"].values

Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_base, y, test_size=0.2, random_state=42
)

model_base = LinearRegression()
model_base.fit(Xb_train, yb_train)

yb_pred = model_base.predict(Xb_test)
rmse_base = np.sqrt(mean_squared_error(yb_test, yb_pred))
r2_base = r2_score(yb_test, yb_pred)

print("Baseline slope:", model_base.coef_[0])
print("Baseline intercept:", model_base.intercept_)
print("Baseline RMSE:", rmse_base)
print("Baseline R2:", r2_base)

# Q: The RMSE is in degrees Celsius. In plain English, how far off is a typical prediction?
# A: Around 2.7 C off. If the real high is 30, the model might say 27 or 33.

# --- Task 5: Full Model ---

feature_cols = ["temperature_2m_min", "precipitation_sum",
                "wind_speed_10m_max", "is_summer"]
X = df[feature_cols].values
y = df["temperature_2m_max"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train, y_train)

y_pred = model_full.predict(X_test)
rmse_full = np.sqrt(mean_squared_error(y_test, y_pred))
r2_train = model_full.score(X_train, y_train)
r2_test = model_full.score(X_test, y_test)

print("Full train R2:", r2_train)
print("Full test R2:", r2_test)
print("Full RMSE:", rmse_full)

print("Coefficients:")
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"  {name:22s}: {coef:+.3f}")

# Q: Interpret the coefficients. Which features raise the predicted high, which lower it?
#    Are any signs surprising? Train R2 vs test R2 -- are they close, or is there a gap?
# A: temp_min and is_summer push the high up. Rain and wind pull it down a bit.
#    Signs all make sense. Train and test R2 are close (0.917 vs 0.938), so no overfitting.

# --- Task 6: Evaluate and Summarize ---

plt.figure(figsize=(6, 6))
plt.scatter(y_pred, y_test, alpha=0.5, edgecolor="k")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color="red", linestyle="--", label="perfect prediction")
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted temp_max (C)")
plt.ylabel("Actual temp_max (C)")
plt.legend()
plt.savefig("outputs/predicted_vs_actual_high.png", dpi=100, bbox_inches="tight")
plt.close()

# Q: Does the model struggle more at the high end, the low end, or is the error roughly even?
# A: Roughly even. Points sit close to the diagonal on both cold and hot days.

# --- Summary ---
# Data: 365 days of weather in 2025. Test set: 73 days.
# Full model: RMSE ~2.4 C, R2 ~0.94. A typical miss is about 2.5 degrees.
# Biggest effect: temp_min (+1.08). Warmer nights -> warmer days, almost 1 to 1.
#   is_summer adds ~1.4 C on top.
# Surprise: adding three features barely helped. R2 went from 0.92 to 0.94.
#   temp_min alone already explains most of the pattern.