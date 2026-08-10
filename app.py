"""
================================================================================
STREAMLIT WEB APP — HEALTHCARE PREDICTIVE ANALYTICS (DIABETES RISK)
================================================================================
Internship Project | Data Analytics Internship (CodTech Technologies)

This app is a presentation layer only. It reuses the exact preprocessing,
trained models, and results already produced by diabetes_risk_prediction.py.
It does NOT duplicate or re-implement the ML pipeline.

Run with:  streamlit run app.py
================================================================================
"""

import os
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Prediction | Healthcare Predictive Analytics",
    page_icon="🩺",
    layout="wide",
)

NAVY = "#1E2761"
ACCENT = "#3D8BFD"
ICE = "#F4F7FE"

st.markdown(f"""
<style>
    .main {{ background-color: #FFFFFF; }}
    .block-container {{ padding-top: 2rem; }}
    h1, h2, h3 {{ color: {NAVY}; }}
    .stButton>button {{
        background-color: {NAVY}; color: white; font-weight: 600;
        border-radius: 8px; padding: 0.6rem 1.4rem; border: none;
    }}
    .stButton>button:hover {{ background-color: {ACCENT}; color: white; }}
    .metric-card {{
        background-color: {ICE}; border-radius: 10px; padding: 1rem 1.2rem;
        border-left: 5px solid {ACCENT};
    }}
    .disclaimer {{
        background-color: #FFF4E5; border-left: 5px solid #F9A03F;
        padding: 0.9rem 1.1rem; border-radius: 8px; font-size: 0.92rem;
    }}
</style>
""", unsafe_allow_html=True)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")


# ------------------------------------------------------------------------------
# LOAD EXISTING PROJECT ARTIFACTS (cached — trains/loads only once)
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """
    Loads trained models + scaler + deployment metadata saved by
    diabetes_risk_prediction.py. If they don't exist yet (first-ever run),
    it runs the existing pipeline script ONCE to generate them — it never
    duplicates the ML logic, it just calls your existing script.
    """
    import joblib

    artifacts_ready = all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in
                           ["trained_models.joblib", "scaler.joblib", "deployment_meta.json"])

    if not artifacts_ready:
        with st.spinner("First-time setup: running the existing ML pipeline once to train and save models..."):
            subprocess.run([sys.executable, "diabetes_risk_prediction.py"], cwd=PROJECT_DIR, check=True)

    models = joblib.load(os.path.join(MODELS_DIR, "trained_models.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    with open(os.path.join(MODELS_DIR, "deployment_meta.json")) as f:
        meta = json.load(f)
    return models, scaler, meta


@st.cache_data(show_spinner=False)
def load_results():
    with open(os.path.join(PROJECT_DIR, "results.json")) as f:
        results = json.load(f)
    metrics_df = pd.read_csv(os.path.join(PROJECT_DIR, "model_comparison_metrics.csv"), index_col=0)
    fi_df = pd.read_csv(os.path.join(PROJECT_DIR, "feature_importance.csv"), index_col=0)
    return results, metrics_df, fi_df


models, scaler, meta = load_artifacts()
results, metrics_df, fi_df = load_results()

BEST_MODEL_NAME = meta["best_model"]
FEATURE_NAMES = meta["feature_names"]
SCALED_MODELS = meta["scaled_models"]
IMPUTE_MEDIANS = meta["impute_medians"]
ZERO_INVALID_COLS = meta["zero_invalid_cols"]
best_model = models[BEST_MODEL_NAME]


def predict_risk(input_dict):
    """Applies the SAME cleaning + scaling logic as diabetes_risk_prediction.py,
    then predicts using the best model identified by the existing pipeline."""
    row = {f: input_dict[f] for f in FEATURE_NAMES}

    # Same rule as training: biologically-invalid zeros -> median from training data
    for col in ZERO_INVALID_COLS:
        if row[col] == 0:
            row[col] = IMPUTE_MEDIANS[col]

    X_input = pd.DataFrame([row], columns=FEATURE_NAMES)

    if BEST_MODEL_NAME in SCALED_MODELS:
        X_input = scaler.transform(X_input)

    pred = best_model.predict(X_input)[0]
    proba = best_model.predict_proba(X_input)[0][1]
    return int(pred), float(proba)


# ==============================================================================
# HEADER
# ==============================================================================
st.markdown(f"""
<div style="background-color:{NAVY}; padding:1.8rem 2rem; border-radius:12px; margin-bottom:1.5rem;">
    <h1 style="color:white; margin:0;">🩺 Healthcare Predictive Analytics</h1>
    <p style="color:{ICE}; font-size:1.05rem; margin-top:0.3rem;">
        Diabetes Risk Prediction Using Machine Learning &nbsp;|&nbsp; Data Analytics Internship — CodTech Technologies
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📌 Navigation")
    page = st.radio("Go to", [
        "🔮 Predict Diabetes Risk",
        "📊 Model Performance",
        "🧬 Feature Importance",
        "ℹ️ About the Project",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 🏆 Best Model")
    st.success(f"**{BEST_MODEL_NAME}**")
    best_metrics = results["model_results"][BEST_MODEL_NAME]
    st.metric("Accuracy", f"{best_metrics['accuracy']*100:.1f}%")
    st.metric("ROC-AUC", f"{best_metrics['roc_auc']:.3f}")

    st.markdown("---")
    st.caption("Built on the existing project pipeline — models are trained once and reused, not retrained on every page load.")


# ==============================================================================
# PAGE 1 — PREDICTION
# ==============================================================================
if page == "🔮 Predict Diabetes Risk":
    st.subheader("Enter Patient Clinical Measurements")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        insulin = st.number_input("Insulin (mu U/ml)", min_value=0.0, max_value=900.0, value=80.0, step=1.0)
    with col2:
        glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
        bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    with col3:
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.47, step=0.01)
    with col4:
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=33, step=1)

    st.write("")
    predict_clicked = st.button("🔍 Predict Diabetes Risk", use_container_width=False)

    if predict_clicked:
        input_dict = {
            "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": dpf, "Age": age,
        }
        pred, proba = predict_risk(input_dict)

        st.write("")
        res_col1, res_col2 = st.columns([1, 1.4])
        with res_col1:
            if pred == 1:
                st.error(f"### ⚠️ Higher Diabetes Risk Predicted")
            else:
                st.success(f"### ✅ Lower Diabetes Risk Predicted")
            st.metric("Estimated Risk Probability", f"{proba*100:.1f}%")
            st.progress(min(max(proba, 0.0), 1.0))
            st.caption(f"Prediction made using **{BEST_MODEL_NAME}** — the best-performing model in this project (ROC-AUC {best_metrics['roc_auc']:.3f}).")

        with res_col2:
            st.markdown("#### Input Summary")
            st.dataframe(pd.DataFrame([input_dict]).T.rename(columns={0: "Value"}), use_container_width=True)

        st.markdown("""
        <div class="disclaimer">
        ⚠️ <b>This is an ML demonstration, not a medical diagnosis.</b> This tool illustrates a data analytics
        internship project and must not be used for real clinical decision-making. Always consult a licensed
        medical professional for actual diagnosis or treatment.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 2 — MODEL PERFORMANCE
# ==============================================================================
elif page == "📊 Model Performance":
    st.subheader("Model Performance Comparison")
    st.caption("All four models below were trained and evaluated in the existing project pipeline (diabetes_risk_prediction.py).")

    display_df = metrics_df.copy()
    display_df.index.name = "Model"
    st.dataframe(
        display_df.style.highlight_max(subset=["accuracy", "precision", "recall", "f1_score", "roc_auc"], color="#CADCFC"),
        use_container_width=True,
    )

    st.markdown(f"🏆 **Best Model (by ROC-AUC): `{BEST_MODEL_NAME}`**")

    c1, c2 = st.columns(2)
    with c1:
        st.image(os.path.join(PLOTS_DIR, "04_model_comparison.png"), caption="Metric comparison across all models", use_container_width=True)
    with c2:
        st.image(os.path.join(PLOTS_DIR, "05_roc_curves.png"), caption="ROC curves — all models", use_container_width=True)

    st.markdown("#### Confusion Matrices")
    cm_files = {
        "Logistic Regression": "cm_logistic_regression.png",
        "Random Forest": "cm_random_forest.png",
        "Support Vector Machine": "cm_support_vector_machine.png",
        "XGBoost": "cm_xgboost.png",
    }
    cols = st.columns(4)
    for (name, fname), col in zip(cm_files.items(), cols):
        with col:
            st.image(os.path.join(PLOTS_DIR, fname), caption=name, use_container_width=True)


# ==============================================================================
# PAGE 3 — FEATURE IMPORTANCE
# ==============================================================================
elif page == "🧬 Feature Importance":
    st.subheader("Feature Importance Analysis")
    st.caption("Computed three ways in the existing pipeline: Random Forest, XGBoost, and Logistic Regression coefficients.")

    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.dataframe(fi_df.style.background_gradient(cmap="Blues", subset=["RandomForest"]), use_container_width=True)
        top5 = results["top_5_features_rf"]
        st.markdown("**Top predictors (Random Forest):**")
        for feat, val in top5.items():
            st.write(f"- **{feat}**: {val:.3f}")
    with c2:
        st.image(os.path.join(PLOTS_DIR, "06_feature_importance.png"), use_container_width=True)

    st.info("Glucose, BMI, and Age consistently rank highest across all three methods — matching established clinical knowledge on diabetes risk factors.")


# ==============================================================================
# PAGE 4 — ABOUT
# ==============================================================================
elif page == "ℹ️ About the Project":
    st.subheader("About This Project")
    st.markdown("""
This project applies supervised machine learning classification to the **Pima Indians Diabetes Database**
to predict whether a patient is at risk of diabetes based on routine clinical measurements. It was built
as part of a **Data Analytics Internship at CodTech Technologies**, covering the full pipeline: data
cleaning, normalization, exploratory data analysis, model training, evaluation, and feature importance
analysis.
""")

    st.markdown("#### 📂 Dataset")
    st.markdown(f"""
- **Source:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
- **Official repository:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/34/diabetes)
- **Official mirror:** [Kaggle (uciml)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Records:** {results['dataset_shape_raw'][0]} patients, {results['dataset_shape_raw'][1]-1} clinical features
- All values are real, individually-measured patient records — not synthetic or predicted data.
""")

    st.markdown("#### 🤖 ML Models")
    st.markdown("""
Four classifiers were trained and compared: **Logistic Regression**, **Random Forest**,
**Support Vector Machine (RBF kernel)**, and **XGBoost**. Each was evaluated with Accuracy, Precision,
Recall, F1-Score, ROC-AUC, and 5-fold cross-validation.
""")

    st.markdown("#### 🔒 Ethical & Privacy Considerations")
    st.markdown("""
- **De-identified data only** — the dataset contains no patient names or identifiers of any kind.
- **Local processing** — all computation runs locally; no patient record is sent to a third party.
- **Aggregate reporting** — every result shown is a group-level model statistic, never a single traceable patient.
- **Bias awareness** — this dataset represents a specific population (Pima Indian heritage, female, age 21+);
  findings should not be generalized to other populations without re-validation.
- **Decision-support only** — this model is a screening aid, not a diagnostic tool, and must never replace
  a licensed clinician's judgment.
- **Regulatory compliance** — any real-world deployment would require informed consent, ethics/IRB approval,
  and compliance with applicable health-data law (HIPAA, India's DPDP Act 2023, GDPR).
""")

    st.markdown("---")
    st.caption("Streamlit interface built on top of the existing project pipeline (diabetes_risk_prediction.py) — no ML logic was duplicated.")
