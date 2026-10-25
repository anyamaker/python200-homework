"""Tests for the WeatherClassifier component."""

import joblib
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from weather_model import WeatherClassifier, Prediction


@pytest.fixture
def model_path(tmp_path):
    """Build a tiny KNN model on hand-crafted data and save it to tmp_path."""
    X_train = [
        [22.0, 12.0, 0.0, 5.0],     # good
        [18.0, 8.0, 0.0, 8.0],      # good
        [25.0, 15.0, 0.0, 6.0],     # good
        [40.0, 25.0, 0.0, 5.0],     # skip (too hot)
        [5.0, -2.0, 10.0, 40.0],    # skip (cold + rain + wind)
        [20.0, 10.0, 15.0, 10.0],   # skip (rain)
    ]
    y_train = [1, 1, 1, 0, 0, 0]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=1)),
    ])
    pipe.fit(X_train, y_train)

    path = tmp_path / "tiny_model.pkl"
    joblib.dump(pipe, path)
    return path


def _day(t_max, t_min, precip, wind):
    return {
        "temperature_2m_max": t_max,
        "temperature_2m_min": t_min,
        "precipitation_sum": precip,
        "wind_speed_10m_max": wind,
    }


@pytest.mark.parametrize("day, expected_label", [
    (_day(22.0, 12.0, 0.0, 5.0), "good"),
    (_day(40.0, 25.0, 0.0, 5.0), "skip"),
])
def test_label_for_known_day(model_path, day, expected_label):
    clf = WeatherClassifier(str(model_path))
    result = clf.predict([day])
    assert len(result) == 1
    assert isinstance(result[0], Prediction)
    assert result[0].label == expected_label


def test_probability_is_between_0_and_1(model_path):
    clf = WeatherClassifier(str(model_path))
    result = clf.predict([_day(22.0, 12.0, 0.0, 5.0)])
    assert 0.0 <= result[0].probability <= 1.0


def test_multiple_days_return_in_order(model_path):
    clf = WeatherClassifier(str(model_path))
    days = [
        _day(22.0, 12.0, 0.0, 5.0),   # good
        _day(40.0, 25.0, 0.0, 5.0),   # skip
        _day(18.0, 8.0, 0.0, 8.0),    # good
    ]
    result = clf.predict(days)
    assert len(result) == 3
    assert result[0].label == "good"
    assert result[1].label == "skip"
    assert result[2].label == "good"


def test_empty_list_returns_empty_list(model_path):
    clf = WeatherClassifier(str(model_path))
    assert clf.predict([]) == []


def test_missing_feature_raises(model_path):
    clf = WeatherClassifier(str(model_path))
    bad_day = {
        "temperature_2m_max": 22.0,
        "temperature_2m_min": 12.0,
        # precipitation_sum is missing on purpose
        "wind_speed_10m_max": 5.0,
    }
    with pytest.raises(ValueError, match="precipitation_sum"):
        clf.predict([bad_day])


def test_missing_model_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.pkl"
    with pytest.raises(FileNotFoundError, match="not found"):
        WeatherClassifier(str(missing))