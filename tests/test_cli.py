import os
import subprocess
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_train_cli_defaults():
    result = subprocess.run(
        [sys.executable, "train.py", "--n-samples", "100", "--no-grid-search"],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "Best model" in result.stdout


def test_train_cli_with_model():
    result = subprocess.run(
        [
            sys.executable, "train.py",
            "--n-samples", "100",
            "--no-grid-search",
            "--model", "Random Forest",
            "--output", "/tmp/test_rf_model.joblib",
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "Best model" in result.stdout
