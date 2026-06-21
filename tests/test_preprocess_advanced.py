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


def test_transform_after_fit(sample_data):
    p = Preprocessor(scale=True)
    p.fit_transform(sample_data)
    X2, y2 = p.transform(sample_data)
    assert not np.any(np.isnan(X2))
    assert X2.shape[0] == len(sample_data)


def test_transform_raises_on_new_labels(sample_data):
    p = Preprocessor()
    p.fit_transform(sample_data)
    df_new = sample_data.copy()
    df_new["class_label"] = 99
    with pytest.raises(Exception):
        p.transform(df_new)


def test_fit_transform_all_nan_label():
    df = pd.DataFrame({
        "u": [1.0, 2.0], "g": [1.0, 2.0], "r": [1.0, 2.0], "i": [1.0, 2.0],
        "z": [1.0, 2.0], "redshift": [0.1, 0.2], "spec_index": [0.0, 0.1],
        "petro_r50": [1.0, 2.0], "petro_r90": [2.0, 4.0],
        "class_label": [float("nan"), float("nan")],
    })
    p = Preprocessor()
    X, y = p.fit_transform(df)
    assert not np.any(np.isnan(X))
    assert y.shape[0] == 2


def test_tsne_reduction(sample_data):
    p = Preprocessor(scale=True, use_tsne=True, n_components=2)
    X, y = p.fit_transform(sample_data)
    assert X.shape[1] == 2


def test_preprocessor_keeps_feature_count_without_reduction(sample_data):
    p = Preprocessor(scale=True, use_pca=False, use_tsne=False)
    X, y = p.fit_transform(sample_data)
    assert X.shape[1] == 9
