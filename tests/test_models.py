import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from models import get_models, train_model


@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=200,
        n_features=9,
        n_informative=6,
        n_redundant=2,
        n_classes=3,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


def test_get_models_returns_all_classifiers():
    models = get_models()
    expected_names = ["Random Forest", "XGBoost", "Logistic Regression", "Neural Network"]
    for name in expected_names:
        assert name in models, f"Missing model: {name}"
    assert len(models) == 4, f"Expected 4 models, got {len(models)}"


def test_get_models_return_callable_classifiers():
    models = get_models()
    for name, model in models.items():
        assert hasattr(model, "fit"), f"{name} has no fit method"
        assert hasattr(model, "predict"), f"{name} has no predict method"


def test_train_model_returns_trained_classifier(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    models = get_models()
    for name, model in models.items():
        trained = train_model(model, X_train, y_train)
        preds = trained.predict(X_test)
        assert preds.shape[0] == X_test.shape[0], (
            f"{name}: prediction shape mismatch"
        )
        assert set(np.unique(preds)).issubset({0, 1, 2}), (
            f"{name}: unexpected prediction classes"
        )


def test_train_model_accepts_hyperparameter_grid(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    model = get_models()["Random Forest"]
    param_grid = {"n_estimators": [10, 50]}
    trained = train_model(model, X_train, y_train, param_grid=param_grid)
    preds = trained.predict(X_test)
    assert preds.shape[0] == X_test.shape[0]


def test_all_models_achieve_reasonable_accuracy(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    models = get_models()
    for name, model in models.items():
        trained = train_model(model, X_train, y_train)
        score = trained.score(X_test, y_test)
        assert score > 0.3, (
            f"{name}: accuracy {score:.3f} < 0.3 (worse than random)"
        )
