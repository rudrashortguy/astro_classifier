import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from data_loader import FEATURE_NAMES, generate_synthetic_data
from evaluate import plot_feature_importance


def test_plot_feature_importance_logistic_regression():
    df = generate_synthetic_data(n_samples=200)
    X = df[FEATURE_NAMES].values
    y = df["class_label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    path = plot_feature_importance(model, FEATURE_NAMES, save_path="/tmp/test_fi_lr.png")
    assert os.path.exists(path)
    os.remove(path)
