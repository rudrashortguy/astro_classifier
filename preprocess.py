import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler

from data_loader import FEATURE_NAMES, TARGET_NAME


class Preprocessor:
    def __init__(
        self,
        scale: bool = True,
        use_pca: bool = False,
        n_components: int | None = None,
        use_tsne: bool = False,
    ):
        self.scale = scale
        self.use_pca = use_pca
        self.n_components = n_components
        self.use_tsne = use_tsne
        self.scaler = StandardScaler() if scale else None
        self.pca = PCA(n_components=n_components) if use_pca else None
        n_tsne = min(n_components or 2, 3)
        self.tsne = TSNE(n_components=n_tsne, random_state=42) if use_tsne else None
        self.label_encoder = LabelEncoder()

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = df[FEATURE_NAMES].copy().astype(float)
        y = df[TARGET_NAME].copy()

        X = X.fillna(X.mean())
        y = y.fillna(y.mode().iloc[0] if not y.mode().empty else 0)

        y_encoded = self.label_encoder.fit_transform(y)

        if self.scaler:
            X = self.scaler.fit_transform(X)
        else:
            X = X.values

        if self.pca:
            X = self.pca.fit_transform(X)
        elif self.tsne:
            X = self.tsne.fit_transform(X)

        return X.astype(np.float64), y_encoded.astype(np.int64)

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = df[FEATURE_NAMES].copy().astype(float)
        y = df[TARGET_NAME].copy()

        X = X.fillna(X.mean())
        y = y.fillna(y.mode().iloc[0] if not y.mode().empty else 0)

        y_encoded = self.label_encoder.transform(y)

        if self.scaler:
            X = self.scaler.transform(X)
        else:
            X = X.values

        if self.pca:
            X = self.pca.transform(X)
        elif self.tsne:
            X = self.tsne.fit_transform(X)

        return X.astype(np.float64), y_encoded.astype(np.int64)
