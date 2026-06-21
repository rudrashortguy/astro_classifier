import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # noqa: F401
from sklearn.metrics import (
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
)

from data_loader import CLASS_MAP

sns.set_style("whitegrid")


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path="confusion_matrix.png"):
    if class_names is None:
        class_names = list(CLASS_MAP.values())
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    return save_path


def plot_roc_curves(y_true, y_score, n_classes=3, save_path="roc_curves.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    class_names = list(CLASS_MAP.values())
    for i in range(n_classes):
        y_bin = (y_true == i).astype(int)
        RocCurveDisplay.from_predictions(
            y_bin, y_score[:, i], name=class_names[i], ax=ax
        )
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    return save_path


def plot_feature_importance(model, feature_names, save_path="feature_importance.png"):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        return None

    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importances)), importances[indices])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_title("Feature Importance")
    ax.set_ylabel("Importance")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    return save_path


def generate_report(y_true, y_pred, class_names=None) -> str:
    if class_names is None:
        class_names = list(CLASS_MAP.values())
    return classification_report(y_true, y_pred, target_names=class_names)


def evaluate_model(model, X_test, y_test, feature_names, save_prefix="eval"):
    y_pred = model.predict(X_test)

    cm_path = plot_confusion_matrix(y_test, y_pred, save_path=f"{save_prefix}_confusion_matrix.png")

    roc_path = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
        roc_path = plot_roc_curves(y_test, y_score, save_path=f"{save_prefix}_roc_curves.png")

    fi_path = plot_feature_importance(
        model, feature_names, save_path=f"{save_prefix}_feature_importance.png"
    )

    report = generate_report(y_test, y_pred)

    return {
        "confusion_matrix_path": cm_path,
        "roc_curves_path": roc_path,
        "feature_importance_path": fi_path,
        "classification_report": report,
        "accuracy": float(model.score(X_test, y_test)),
    }
