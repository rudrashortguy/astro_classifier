from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def get_models() -> dict:
    return {
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, verbosity=0),
        "Logistic Regression": LogisticRegression(
            random_state=42, max_iter=1000
        ),
        "Neural Network": MLPClassifier(
            random_state=42, max_iter=500, hidden_layer_sizes=(64, 32)
        ),
    }


def get_param_grids() -> dict:
    return {
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
        },
        "XGBoost": {
            "n_estimators": [50, 100],
            "max_depth": [3, 6],
            "learning_rate": [0.01, 0.1],
        },
        "Logistic Regression": {
            "C": [0.1, 1.0, 10.0],
            "solver": ["lbfgs", "newton-cg"],
        },
        "Neural Network": {
            "hidden_layer_sizes": [(64,), (64, 32)],
            "learning_rate_init": [0.001, 0.01],
        },
    }


def train_model(
    model,
    X_train,
    y_train,
    param_grid: dict | None = None,
    cv: int = 3,
):
    if param_grid:
        gs = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        return gs.best_estimator_
    model.fit(X_train, y_train)
    return model
