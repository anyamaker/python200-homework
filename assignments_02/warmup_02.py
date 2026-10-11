# --- scikit-learn API ---
# Q1

import numpy as np
from sklearn.linear_model import LinearRegression

temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

model = LinearRegression()
model.fit(temp_min, temp_max)

predictions = model.predict(np.array([[6], [18]]))

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Prediction for low=6:", predictions[0])
print("Prediction for low=18:", predictions[1])

# Q2

x = np.array([10, 20, 30, 40, 50])
print("Shape of x:", x.shape)

x_2d = x.reshape(-1, 1)
print("Shape of x_2d:", x_2d.shape)

# Q: Why does scikit-learn need X to be 2D?
# A: It treats X as a table: rows are days, columns are features. A 1D array has no columns, so it can't tell features apart.


# --- Linear Regression ---

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(42)
num_days = 120
temp_min = rng.uniform(-5, 22, num_days)
is_rainy = rng.integers(0, 2, num_days).astype(float)
temp_max = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)

# Q1

plt.figure(figsize=(7, 5))
plt.scatter(temp_min, temp_max, c=is_rainy, cmap="coolwarm", alpha=0.7)
plt.title("Daily High vs Low")
plt.xlabel("temp_min (C)")
plt.ylabel("temp_max (C)")
plt.colorbar(label="is_rainy")
plt.savefig("outputs/high_vs_low.png", dpi=100, bbox_inches="tight")
plt.close()

# Q: What do you see? Are there two visible groups? What does that suggest about is_rainy?
# A: Two parallel bands. Blue (dry) sits higher, red (rainy) sits lower. Rain drops the high by about 4 degrees.

# Q2

X = temp_min.reshape(-1, 1)
y = temp_max

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Q3

model = LinearRegression()
model.fit(X_train, y_train)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model.score(X_test, y_test)
print("RMSE:", rmse)
print("R2:", r2)

# Q: Interpret the slope in plain English. What does it mean for the daily high?
# A: Each extra degree of temp_min adds about 1 degree to the predicted high.

# Q4

X_full = np.column_stack([temp_min, is_rainy])
Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(Xf_train, yf_train)

r2_full = model_full.score(Xf_test, yf_test)
print("Test R2 with is_rainy:", r2_full)
print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])

# Q: Interpret the is_rainy coefficient. What does it represent in practical terms?
# A: It's negative, around -3. On a rainy day the predicted high is a few degrees lower than on a dry day with the same temp_min.

# Q5

yf_pred = model_full.predict(Xf_test)

plt.figure(figsize=(6, 6))
plt.scatter(yf_pred, yf_test, alpha=0.7, edgecolor="k")
plt.plot([yf_test.min(), yf_test.max()], [yf_test.min(), yf_test.max()],
         color="red", linestyle="--", label="perfect prediction")
plt.title("Predicted vs Actual")
plt.xlabel("Predicted temp_max (C)")
plt.ylabel("Actual temp_max (C)")
plt.legend()
plt.savefig("outputs/predicted_vs_actual_warmup.png", dpi=100, bbox_inches="tight")
plt.close()

# Q: What does it mean when a point falls above the diagonal? What about below?
# A: Above means the real value was higher than the model said. Below means it was lower.