import os
import tempfile

import joblib
import pandas as pd
import streamlit as st

from data_loader import CLASS_MAP, FEATURE_NAMES, TARGET_NAME, generate_synthetic_data
from evaluate import evaluate_model
from models import get_models, get_param_grids, train_model
from preprocess import Preprocessor

st.set_page_config(page_title="Astro Classifier", layout="wide")

st.title("🔭 Astrophysical Data Classifier")
st.markdown(
    "Classify celestial objects as **Stars**, **Galaxies**, or **Quasars**"
    " using SDSS photometric data."
)

tab1, tab2, tab3 = st.tabs(["Train & Evaluate", "Predict", "About"])

with tab1:
    st.header("Model Training")

    col_left, col_right = st.columns(2)

    with col_left:
        data_option = st.radio("Data source", ["Use synthetic data", "Upload CSV"], index=0)
        n_samples = st.slider("Number of synthetic samples", 500, 10000, 5000, step=500)

    with col_right:
        model_names = list(get_models().keys())
        selected_model = st.selectbox("Select model", ["All"] + model_names)
        use_grid = st.checkbox("Hyperparameter search", value=True)
        use_pca = st.checkbox("Apply PCA", value=False)
        n_components = st.slider("PCA components", 2, 9, 5) if use_pca else None

    uploaded_file = None
    if data_option == "Upload CSV":
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")

    if st.button("Train Model", type="primary"):
        with st.spinner("Training in progress..."):
            temp_dir = tempfile.mkdtemp()

            if uploaded_file:
                df = pd.read_csv(uploaded_file)
            else:
                df = generate_synthetic_data(n_samples=n_samples)

            preprocessor = Preprocessor(scale=True, use_pca=use_pca, n_components=n_components)
            X, y = preprocessor.fit_transform(df)
            feature_names = [f"PC{i+1}" for i in range(X.shape[1])] if use_pca else FEATURE_NAMES

            models = get_models()
            param_grids = get_param_grids() if use_grid else {}
            names_to_train = [selected_model] if selected_model != "All" else list(models.keys())

            results = {}
            best_accuracy = -1.0
            best_model = None
            best_name = ""

            for name in names_to_train:
                model = models[name]
                grid = param_grids.get(name) if use_grid else None
                grid = grid if grid else None

                trained = train_model(model, X, y, param_grid=grid)
                eval_results = evaluate_model(
                    trained, X, y, feature_names,
                    save_prefix=os.path.join(temp_dir, f"{name.lower().replace(' ', '_')}")
                )
                results[name] = eval_results

                if eval_results["accuracy"] > best_accuracy:
                    best_accuracy = eval_results["accuracy"]
                    best_model = trained
                    best_name = name

            artifact = {
                "model": best_model,
                "model_name": best_name,
                "preprocessor": preprocessor,
                "test_accuracy": best_accuracy,
            }
            model_path = os.path.join(temp_dir, "best_model.joblib")
            joblib.dump(artifact, model_path)
            st.session_state["model_path"] = model_path
            st.session_state["trained"] = True

            for name, r in results.items():
                expanded = name == best_name
                with st.expander(f"{name} — Accuracy: {r['accuracy']:.4f}", expanded=expanded):
                    col1, col2 = st.columns(2)
                    with col1:
                        if os.path.exists(r["confusion_matrix_path"]):
                            st.image(r["confusion_matrix_path"], caption="Confusion Matrix")
                    with col2:
                        if r["roc_curves_path"] and os.path.exists(r["roc_curves_path"]):
                            st.image(r["roc_curves_path"], caption="ROC Curves")
                    fi_path = r["feature_importance_path"]
                    if fi_path and os.path.exists(fi_path):
                        st.image(r["feature_importance_path"], caption="Feature Importance")
                    st.text(r["classification_report"])

            st.success(
                f"Training complete! Best model: **{best_name}** "
                f"(Accuracy: **{best_accuracy:.4f}**)"
            )

with tab2:
    st.header("Make Predictions")

    trained = st.session_state.get("trained")
    model_path = st.session_state.get("model_path", "")
    if not trained and not os.path.exists(model_path):
        st.warning("Please train a model in the 'Train & Evaluate' tab first.")
    else:
        model_path = st.session_state.get("model_path", "")
        if model_path and os.path.exists(model_path):
            artifact = joblib.load(model_path)
            model = artifact["model"]
            preprocessor = artifact["preprocessor"]

            pred_options = ["Enter feature values", "Upload CSV"]
            pred_source = st.radio("Prediction input", pred_options, index=0)

            if pred_source == "Enter feature values":
                st.subheader("Enter photometric magnitudes")
                cols = st.columns(5)
                input_values = {}
                for i, feat in enumerate(FEATURE_NAMES):
                    with cols[i % 5]:
                        default_val = 20.0 if feat in ["u", "g", "r", "i", "z"] else 0.5
                        input_values[feat] = st.number_input(
                            feat, value=default_val, format="%.4f",
                            key=f"pred_{feat}"
                        )

                if st.button("Classify"):
                    input_df = pd.DataFrame([input_values])
                    input_df[TARGET_NAME] = 0
                    X, _ = preprocessor.transform(input_df)
                    pred = model.predict(X)
                    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
                    label = CLASS_MAP.get(int(pred[0]), "Unknown")
                    st.success(f"Prediction: **{label}**")
                    if proba is not None:
                        st.write("Probabilities:")
                        for i, name in CLASS_MAP.items():
                            st.write(f"- {name}: {proba[0][i]:.4f}")
            else:
                pred_file = st.file_uploader(
                    "Upload CSV for prediction", type="csv", key="pred_csv"
                )
                if pred_file is not None:
                    df_pred = pd.read_csv(pred_file)
                    if TARGET_NAME not in df_pred.columns:
                        df_pred[TARGET_NAME] = 0
                    X_pred, _ = preprocessor.transform(df_pred)
                    preds = model.predict(X_pred)
                    df_pred["prediction"] = [CLASS_MAP.get(p, "Unknown") for p in preds]
                    st.dataframe(df_pred)
                    csv = df_pred.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions", csv, "predictions.csv", "text/csv")

with tab3:
    st.header("About")
    st.markdown("""
    This app classifies celestial objects using photometric data from the
    Sloan Digital Sky Survey (SDSS).

    **Features used:**
    - **u, g, r, i, z** — Ultraviolet to near-infrared magnitudes
    - **redshift** — Cosmological redshift
    - **spec_index** — Spectral index
    - **petro_r50, petro_r90** — Petrosian radii

    **Available classifiers:**
    - Random Forest
    - XGBoost
    - Logistic Regression
    - Neural Network (MLP)

    **Data:** Uses synthetic SDSS-like data or uploaded CSV files.
    """)
