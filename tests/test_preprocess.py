import numpy as np
import pandas as pd
import pytest

from preprocess import Preprocessor


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "u": rng.uniform(15, 25, n),
            "g": rng.uniform(14, 24, n),
            "r": rng.uniform(13, 23, n),
            "i": rng.uniform(12, 22, n),
            "z": rng.uniform(11, 21, n),
            "redshift": rng.uniform(0, 3, n),
            "spec_index": rng.uniform(-1, 1, n),
            "petro_r50": rng.uniform(0, 10, n),
            "petro_r90": rng.uniform(0, 20, n),
            "class_label": rng.integers(0, 3, n),
        }
    )
    return df


@pytest.fixture
def data_with_missing():
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "u": rng.uniform(15, 25, n),
            "g": rng.uniform(14, 24, n),
            "r": rng.uniform(13, 23, n),
            "i": rng.uniform(12, 22, n),
            "z": rng.uniform(11, 21, n),
            "redshift": rng.uniform(0, 3, n),
            "spec_index": rng.uniform(-1, 1, n),
            "petro_r50": rng.uniform(0, 10, n),
            "petro_r90": rng.uniform(0, 20, n),
            "class_label": rng.integers(0, 3, n),
        }
    )
    df.loc[0:4, "u"] = np.nan
    df.loc[2:6, "redshift"] = np.nan
    return df


def test_preprocess_returns_no_nan(sample_data):
    p = Preprocessor()
    X, y = p.fit_transform(sample_data)
    assert not np.any(np.isnan(X)), "Preprocessed features contain NaN values"
    assert not np.any(np.isnan(y)), "Labels contain NaN values"


def test_preprocess_returns_correct_shape(sample_data):
    p = Preprocessor()
    X, y = p.fit_transform(sample_data)
    assert X.shape[0] == len(sample_data), "Number of samples changed"
    assert X.shape[1] == 9, "Expected 9 feature columns"
    assert y.shape[0] == len(sample_data), "Label count mismatch"


def test_preprocess_handles_missing_values(data_with_missing):
    p = Preprocessor()
    X, y = p.fit_transform(data_with_missing)
    assert not np.any(np.isnan(X)), "Missing values remain after preprocessing"
    assert X.shape[0] == len(data_with_missing), "Rows dropped unexpectedly"


def test_preprocess_scales_features(sample_data):
    p = Preprocessor(scale=True)
    X, y = p.fit_transform(sample_data)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    assert np.allclose(means, 0, atol=1e-10), "Scaled features not centered"
    assert np.allclose(stds, 1, atol=1e-10), "Scaled features not unit variance"


def test_preprocess_pca_reduces_dimensions(sample_data):
    p = Preprocessor(scale=True, use_pca=True, n_components=3)
    X, y = p.fit_transform(sample_data)
    assert X.shape[1] == 3, f"PCA reduction to 3 components, got {X.shape[1]}"


def test_preprocess_without_scaling(sample_data):
    p = Preprocessor(scale=False)
    X_before = sample_data.drop(columns=["class_label"]).values.astype(float)
    X, y = p.fit_transform(sample_data)
    # Values should be roughly the same range since no scaling
    assert X.shape == X_before.shape


def test_fit_transform_returns_self(sample_data):
    p = Preprocessor()
    result = p.fit_transform(sample_data)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_label_encoding(sample_data):
    p = Preprocessor()
    X, y = p.fit_transform(sample_data)
    classes = np.unique(y)
    assert set(classes).issubset({0, 1, 2}), f"Unexpected label values: {classes}"
