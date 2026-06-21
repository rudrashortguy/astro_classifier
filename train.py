import joblib
from sklearn.model_selection import cross_val_score, train_test_split

from data_loader import load_sdss_data
from models import get_models, get_param_grids, train_model
from preprocess import Preprocessor


def train_pipeline(
    data_source: str | None = None,
    n_samples: int = 5000,
    test_size: float = 0.2,
    use_param_search: bool = True,
    model_name: str | None = None,
    output_path: str = "astro_classifier_model.joblib",
    use_pca: bool = False,
    n_components: int | None = None,
):

    df = load_sdss_data(filepath=data_source, n_samples=n_samples)
    preprocessor = Preprocessor(scale=True, use_pca=use_pca, n_components=n_components)
    X, y = preprocessor.fit_transform(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    models = get_models()
    param_grids = get_param_grids() if use_param_search else None

    results = {}
    best_model_obj = None
    best_score = -1.0
    best_name = ""

    names_to_train = [model_name] if model_name else list(models.keys())

    for name in names_to_train:
        model = models[name]
        grid = param_grids.get(name) if param_grids else None
        trained = train_model(model, X_train, y_train, param_grid=grid)

        cv_scores = cross_val_score(trained, X_train, y_train, cv=3, scoring="accuracy")
        test_score = trained.score(X_test, y_test)

        results[name] = {
            "model": trained,
            "cv_score_mean": float(cv_scores.mean()),
            "cv_score_std": float(cv_scores.std()),
            "test_score": float(test_score),
        }

        if test_score > best_score:
            best_score = test_score
            best_model_obj = trained
            best_name = name

    artifact = {
        "model": best_model_obj,
        "model_name": best_name,
        "preprocessor": preprocessor,
        "results": results,
        "test_accuracy": best_score,
    }

    joblib.dump(artifact, output_path)

    print(f"Best model: {best_name} (test accuracy: {best_score:.4f})")
    for name, r in results.items():
        print(f"  {name}: CV = {r['cv_score_mean']:.4f} ± {r['cv_score_std']:.4f}, "
              f"Test = {r['test_score']:.4f}")

    return artifact


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Train astrophysical classifier")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV data file")
    parser.add_argument("--n-samples", type=int, default=5000, help="Number of synthetic samples")
    parser.add_argument("--model", type=str, default=None, help="Specific model to train")
    parser.add_argument("--output", type=str, default="astro_classifier_model.joblib")
    parser.add_argument("--no-grid-search", action="store_true", help="Skip hyperparameter search")
    parser.add_argument("--pca", type=int, default=None, help="PCA components")
    args = parser.parse_args()

    train_pipeline(
        data_source=args.data,
        n_samples=args.n_samples,
        use_param_search=not args.no_grid_search,
        model_name=args.model,
        output_path=args.output,
        use_pca=args.pca is not None,
        n_components=args.pca,
    )
