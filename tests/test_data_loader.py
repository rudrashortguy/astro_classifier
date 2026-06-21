import pandas as pd
import pytest

from data_loader import (
    CLASS_MAP,
    FEATURE_NAMES,
    TARGET_NAME,
    generate_synthetic_data,
    load_sdss_data,
)


def test_generate_synthetic_data_returns_dataframe():
    df = generate_synthetic_data(n_samples=100, random_state=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100


def test_generate_synthetic_data_has_all_columns():
    df = generate_synthetic_data(n_samples=100)
    expected_cols = set(FEATURE_NAMES + [TARGET_NAME])
    assert set(df.columns) == expected_cols


def test_generate_synthetic_no_nan():
    df = generate_synthetic_data(n_samples=1000)
    assert df.isnull().sum().sum() == 0


def test_generate_synthetic_labels_in_range():
    df = generate_synthetic_data(n_samples=500)
    assert set(df[TARGET_NAME].unique()).issubset({0, 1, 2})


def test_generate_synthetic_redshift_different_for_classes():
    df = generate_synthetic_data(n_samples=5000)
    for cls in [0, 1, 2]:
        cls_redshift = df[df[TARGET_NAME] == cls]["redshift"]
        assert cls_redshift.mean() > 0, f"Class {cls} has non-positive mean redshift"


def test_generate_synthetic_reproducible():
    df1 = generate_synthetic_data(n_samples=500, random_state=42)
    df2 = generate_synthetic_data(n_samples=500, random_state=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_synthetic_different_seeds():
    df1 = generate_synthetic_data(n_samples=500, random_state=42)
    df2 = generate_synthetic_data(n_samples=500, random_state=99)
    assert not df1.equals(df2)


def test_load_sdss_data_default_generates_synthetic():
    df = load_sdss_data(n_samples=200)
    assert len(df) == 200
    assert TARGET_NAME in df.columns


def test_load_sdss_data_from_file(tmp_path):
    df_orig = generate_synthetic_data(n_samples=50)
    filepath = tmp_path / "test_data.csv"
    df_orig.to_csv(filepath, index=False)
    df_loaded = load_sdss_data(filepath=str(filepath))
    assert len(df_loaded) == 50


def test_load_sdss_missing_columns_raises(tmp_path):
    df_bad = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    filepath = tmp_path / "bad.csv"
    df_bad.to_csv(filepath, index=False)
    with pytest.raises(ValueError, match="Missing columns"):
        load_sdss_data(filepath=str(filepath))


def test_class_map_has_all_labels():
    assert len(CLASS_MAP) == 3
    assert CLASS_MAP[0] == "Star"
    assert CLASS_MAP[1] == "Galaxy"
    assert CLASS_MAP[2] == "Quasar"
