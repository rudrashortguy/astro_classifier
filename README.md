# 🔭 Astrophysical Data Classifier

Classify celestial objects (stars, galaxies, quasars) using machine learning on SDSS photometric data.

## Features

- **4 classifiers**: Random Forest, XGBoost, Logistic Regression, Neural Network (MLP)
- **Hyperparameter tuning** via GridSearchCV with cross-validation
- **Dimensionality reduction**: PCA and t-SNE
- **Comprehensive evaluation**: confusion matrices, ROC curves, feature importance plots
- **Interactive dashboard** built with Streamlit
- **Synthetic data generator** for SDSS-like photometric data (magnitudes in u,g,r,i,z bands, redshift, etc.)
- **46 tests** with 100% code coverage

## Quick Start

```bash
poetry install
poetry run streamlit run app.py
```

## Train from CLI

```bash
poetry run python train.py --n-samples 5000 --model "Random Forest"
```

## Test

```bash
poetry run pytest tests/ --cov=. --cov-config=.coveragerc
```

## Project Structure

```
├── app.py              # Streamlit dashboard
├── data_loader.py      # Synthetic SDSS data generator
├── preprocess.py       # Scaling, PCA, t-SNE, imputation
├── models.py           # Classifier definitions + hyperparameter grids
├── train.py            # Training pipeline with cross-validation
├── evaluate.py         # Confusion matrix, ROC, feature importance plots
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocess.py / test_preprocess_advanced.py / test_preprocess_edge.py
│   ├── test_models.py
│   ├── test_train.py
│   ├── test_evaluate.py / test_evaluate_edge.py
│   └── test_cli.py
└── pyproject.toml
```
