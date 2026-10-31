"""Weather classifier component: load a saved model and predict good/skip."""

from dataclasses import dataclass

import joblib
import pandas as pd


FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]


@dataclass
class Prediction:
    """One prediction for a single day.

    Attributes:
        label: "good" if the day is good for running, "skip" otherwise.
        probability: Probability (0 to 1) that the day is good for running.
    """
    label: str
    probability: float


class WeatherClassifier:
    """Load a saved weather pipeline and predict good/skip for running days."""

    def __init__(self, model_path: str):
        """Load the saved pipeline from model_path.

        Args:
            model_path: Path to a .pkl file containing a fitted scikit-learn pipeline.

        Raises:
            FileNotFoundError: if no file exists at model_path.
        """
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model file not found at {model_path!r}. "
                "Run train_classifier.py first to create it."
            )

    def predict(self, days: list[dict]) -> list[Prediction]:
        """Predict good/skip for each day.

        Args:
            days: List of dicts. Each dict must contain keys
                temperature_2m_max, temperature_2m_min, precipitation_sum,
                wind_speed_10m_max.

        Returns:
            List of Prediction objects, one per day, in the same order.

        Raises:
            ValueError: if any required feature is missing from a day dict.
        """
        if not days:
            return []

        for i, day in enumerate(days):
            missing = [f for f in FEATURES if f not in day]
            if missing:
                raise ValueError(
                    f"Day {i} is missing required features: {missing}"
                )

        X = pd.DataFrame(
            [[day[f] for f in FEATURES] for day in days],
            columns=FEATURES,
        )

        labels = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        return [
            Prediction(
                label="good" if int(lbl) == 1 else "skip",
                probability=float(prob),
            )
            for lbl, prob in zip(labels, probabilities)
        ]


# Breaking predict() to always return "skip" made test_label_for_known_day[day0-good]
# and test_multiple_days_return_in_order fail. Fixed back to the normal label logic.