import os
import tempfile

from models import get_models
from train import train_pipeline


def test_train_pipeline_runs_successfully():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "model.joblib")
        artifact = train_pipeline(
            n_samples=200,
            test_size=0.3,
            use_param_search=False,
            output_path=output_path,
        )
        assert "model" in artifact
        assert "preprocessor" in artifact
        assert "results" in artifact
        assert artifact["model_name"] in get_models()
        assert os.path.exists(output_path)


def test_train_pipeline_with_grid_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "model_grid.joblib")
        artifact = train_pipeline(
            n_samples=200,
            test_size=0.3,
            use_param_search=True,
            output_path=output_path,
        )
        assert artifact["test_accuracy"] >= 0


def test_train_pipeline_with_specific_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "model_rf.joblib")
        artifact = train_pipeline(
            n_samples=200,
            test_size=0.3,
            use_param_search=False,
            model_name="Random Forest",
            output_path=output_path,
        )
        assert artifact["model_name"] == "Random Forest"


def test_train_pipeline_with_pca():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "model_pca.joblib")
        artifact = train_pipeline(
            n_samples=200,
            test_size=0.3,
            use_param_search=False,
            output_path=output_path,
            use_pca=True,
            n_components=3,
        )
        assert artifact["test_accuracy"] >= 0
        assert os.path.exists(output_path)
