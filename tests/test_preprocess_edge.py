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


def test_transform_no_scaler(sample_data):
    p = Preprocessor(scale=False)
    p.fit_transform(sample_data)
    X2, y2 = p.transform(sample_data)
    assert X2.shape[1] == 9


def test_transform_with_pca(sample_data):
    p = Preprocessor(scale=True, use_pca=True, n_components=3)
    p.fit_transform(sample_data)
    X2, y2 = p.transform(sample_data)
    assert X2.shape[1] == 3


def test_transform_with_tsne(sample_data):
    p = Preprocessor(scale=False, use_tsne=True, n_components=2)
    p.fit_transform(sample_data)
    X2, y2 = p.transform(sample_data)
    assert X2.shape[1] == 2
    assert not np.any(np.isnan(X2))
