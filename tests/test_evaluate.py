import os

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from data_loader import FEATURE_NAMES, generate_synthetic_data
from evaluate import (
    evaluate_model,
    generate_report,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curves,
)


@pytest.fixture
def trained_model_and_data():
    df = generate_synthetic_data(n_samples=500)
    X = df[FEATURE_NAMES].values
    y = df["class_label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = RandomForestClassifier(random_state=42, n_estimators=50)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def test_plot_confusion_matrix_creates_file(trained_model_and_data):
    model, X_test, y_test = trained_model_and_data
    y_pred = model.predict(X_test)
    path = plot_confusion_matrix(y_test, y_pred, save_path="/tmp/test_cm.png")
    assert os.path.exists(path)
    os.remove(path)


def test_plot_roc_curves_creates_file(trained_model_and_data):
    model, X_test, y_test = trained_model_and_data
    y_score = model.predict_proba(X_test)
    path = plot_roc_curves(y_test, y_score, save_path="/tmp/test_roc.png")
    assert os.path.exists(path)
    os.remove(path)


def test_plot_feature_importance_rf_creates_file(trained_model_and_data):
    model, X_test, y_test = trained_model_and_data
    path = plot_feature_importance(model, FEATURE_NAMES, save_path="/tmp/test_fi.png")
    assert os.path.exists(path)
    os.remove(path)


def test_plot_feature_importance_returns_none_for_no_importance():
    class Dummy:
        pass
    dummy = Dummy()
    path = plot_feature_importance(dummy, FEATURE_NAMES)
    assert path is None


def test_generate_report_returns_string(trained_model_and_data):
    model, X_test, y_test = trained_model_and_data
    y_pred = model.predict(X_test)
    report = generate_report(y_test, y_pred)
    assert isinstance(report, str)
    assert "Star" in report
    assert "Galaxy" in report
    assert "Quasar" in report


def test_evaluate_model_returns_all_keys(trained_model_and_data):
    model, X_test, y_test = trained_model_and_data
    result = evaluate_model(model, X_test, y_test, FEATURE_NAMES, save_prefix="/tmp/test_eval")
    assert "confusion_matrix_path" in result
    assert "classification_report" in result
    assert "accuracy" in result
    for key in ["confusion_matrix_path", "roc_curves_path", "feature_importance_path"]:
        if result[key] and os.path.exists(result[key]):
            os.remove(result[key])


def test_roc_curves_with_custom_classes():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 3, 50)
    y_score = rng.uniform(0, 1, (50, 3))
    y_score = y_score / y_score.sum(axis=1, keepdims=True)
    path = plot_roc_curves(y_true, y_score, n_classes=3, save_path="/tmp/test_roc2.png")
    assert os.path.exists(path)
    os.remove(path)
