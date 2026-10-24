import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Preprocessing ---
# Q1

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Mean of scaled training columns:", X_train_scaled.mean(axis=0))

# Q: Why fit the scaler on X_train only?
# A: So the mean and std come from training data only; fitting on all data would leak test information into the scaling.

# Q2

encoder = OneHotEncoder(sparse_output=False)
season_encoded = encoder.fit_transform(df[["season"]])

print("Shape of encoded season:", season_encoded.shape)
print("Categories:", encoder.get_feature_names_out())

# Q: How many columns did the single season column become, and why?
# A: It became 4 columns because season has 4 categories (winter, spring, summer, fall), one column per category.

# Q3

# X with all features INCLUDING season (needed for Q3)
X_all = df[numeric + ["season"]]
y_all = df["good_for_running"]

Xa_train, Xa_test, ya_train, ya_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(), ["season"]),
    ]
)

pipe_lr = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

pipe_lr.fit(Xa_train, ya_train)
print("Pipeline test accuracy:", pipe_lr.score(Xa_test, ya_test))

# Q: What does the pipeline do automatically that you would otherwise do by hand?
# A: It fits the scaler and encoder on X_train, applies the same transformations to X_test,
#    and passes the result to the classifier -- all in one fit/predict call, with no data leakage.

# --- KNN ---
# Q1

pipe_knn = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ]
)
pipe_knn.fit(X_train, y_train)

knn_pred = pipe_knn.predict(X_test)
print("KNN test accuracy:", accuracy_score(y_test, knn_pred))
print("KNN classification report:")
print(classification_report(y_test, knn_pred))

# Q2

k_values = [1, 3, 5, 7, 9, 11, 15, 21]
for k in k_values:
    pipe_k = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    scores = cross_val_score(pipe_k, X_train, y_train, cv=5)
    print(f"k={k:2d}  mean CV accuracy = {scores.mean():.4f}")

# Q: Which k would you choose, and why?
# A: k=11, because it has the highest mean 5-fold CV accuracy (0.9417).
#    k=1 and k=3 score lower, showing the model is sensitive to noise at very small k.
#    k=15 and k=21 drop off, showing the model gets too smooth when it averages over too many neighbors.

# Q3

# KNN without scaling
pipe_knn_raw = Pipeline([
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
pipe_knn_raw.fit(X_train, y_train)
acc_raw = pipe_knn_raw.score(X_test, y_test)

# KNN with scaling
pipe_knn_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
pipe_knn_scaled.fit(X_train, y_train)
acc_scaled = pipe_knn_scaled.score(X_test, y_test)

print("KNN without scaling:", acc_raw)
print("KNN with scaling:   ", acc_scaled)

# Q: Did scaling help, and why does it matter for a distance-based model?
# A: Yes, scaling improved accuracy. KNN measures distances between points;
#    without scaling, features with larger numeric ranges (like precipitation or wind speed)
#    dominate the distance and the model barely notices temperature. Scaling puts every feature on the same footing.

# --- Logistic Regression ---
# Q1

pipe_log = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])
pipe_log.fit(X_train, y_train)

log_pred = pipe_log.predict(X_test)
print("LogReg accuracy: ", accuracy_score(y_test, log_pred))
print("LogReg precision:", precision_score(y_test, log_pred))
print("LogReg recall:   ", recall_score(y_test, log_pred))
print("LogReg F1:       ", f1_score(y_test, log_pred))

# Q2

logreg = pipe_log.named_steps["logreg"]
for name, coef in zip(numeric, logreg.coef_[0]):
    print(f"  {name:22s}: {coef:+.3f}")

# Q: Which features push most strongly toward "skip," and does that match intuition?
# A: The features with the most negative coefficients push toward "skip" (class 0).
#    Precipitation and wind speed are usually the strongest negative -- rain and wind make
#    a day bad for running even when the temperature is fine. This matches intuition.

# Q3

probs = pipe_log.predict_proba(X_test)[:5]
print("Probabilities for first 5 test days (P(good)):")
for i, p in enumerate(probs):
    print(f"  day {i+1}: {p[1]:.2f}")

# Q: What does a probability near 0.5 tell you about the model's confidence for that day?
# A: Near 0.5 means the model is uncertain -- the day is borderline, and the prediction could easily go either way.

# --- Evaluation ---
# Q1

cm = confusion_matrix(y_test, log_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["skip", "good"])
disp.plot(cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.savefig("outputs/logreg_confusion_matrix.png", dpi=100, bbox_inches="tight")
plt.close()

print("Confusion matrix:")
print(cm)

# Q: How many false positives and false negatives? Which error matters more for a running app?
# A: FP = 9 (predicted good, actually skip), FN = 7 (predicted skip, actually good).
#    For a running app, FP is usually worse -- sending someone out in the rain damages trust
#    more than missing one nice day.

# --- joblib ---
# Q1

joblib.dump(pipe_log, "models/warmup_model.pkl")

loaded = joblib.load("models/warmup_model.pkl")
assert (loaded.predict(X_test) == pipe_log.predict(X_test)).all()
print("Predictions match.")

# Q: What would go wrong if you saved only the LogisticRegression step without the scaler,
#    then called predict() on raw data?
# A: The model was trained on scaled data, so raw inputs would be on a different scale.
#    Predictions would be silently wrong -- no error, just bad results.
