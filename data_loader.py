import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "u", "g", "r", "i", "z",
    "redshift", "spec_index", "petro_r50", "petro_r90",
]

TARGET_NAME = "class_label"

CLASS_MAP = {0: "Star", 1: "Galaxy", 2: "Quasar"}

BAND_RANGES = {
    "u": (14.0, 26.0),
    "g": (13.0, 25.0),
    "r": (12.0, 24.0),
    "i": (11.0, 23.0),
    "z": (10.0, 22.0),
}


def generate_synthetic_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    data = {}
    for band, (lo, hi) in BAND_RANGES.items():
        data[band] = rng.uniform(lo, hi, n_samples)

    data["redshift"] = np.zeros(n_samples)
    data["spec_index"] = rng.uniform(-1, 1, n_samples)
    data["petro_r50"] = rng.uniform(0, 10, n_samples)
    data["petro_r90"] = rng.uniform(0, 20, n_samples)

    labels = rng.integers(0, 3, n_samples)

    star_mask = labels == 0
    galaxy_mask = labels == 1
    quasar_mask = labels == 2

    data["redshift"][star_mask] = rng.uniform(0, 0.1, star_mask.sum())
    data["redshift"][galaxy_mask] = rng.uniform(0.02, 0.5, galaxy_mask.sum())
    data["redshift"][quasar_mask] = rng.uniform(0.5, 6.0, quasar_mask.sum())

    for band in BAND_RANGES:
        data[band] = np.where(
            quasar_mask,
            np.clip(
                data[band] + rng.uniform(-12, -6, n_samples),
                BAND_RANGES[band][0],
                BAND_RANGES[band][1],
            ),
            data[band],
        )

    data[TARGET_NAME] = labels

    df = pd.DataFrame(data)
    return df


def load_sdss_data(filepath: str | None = None, n_samples: int = 5000) -> pd.DataFrame:
    if filepath:
        df = pd.read_csv(filepath)
        required = set(FEATURE_NAMES + [TARGET_NAME])
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in data file: {missing}")
        return df
    return generate_synthetic_data(n_samples=n_samples)
