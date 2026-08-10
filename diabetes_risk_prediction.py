"""
================================================================================
HEALTHCARE PREDICTIVE ANALYTICS: DIABETES RISK DETECTION
================================================================================
Internship Project | Data Analytics Internship (CodTech Technologies)

Objective:
    Predict a patient's risk of Type-2 Diabetes using clinical measurements,
    following an ethically-aware, reproducible machine learning workflow.

Dataset:
    Pima Indians Diabetes Dataset (UCI Machine Learning Repository)
    768 patient records | 8 clinical features | 1 binary target (Outcome)
    Source (public, de-identified): UCI ML Repository / mirrored on GitHub
    https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

Pipeline:
    1. Data Loading & Ethical Handling Notes
    2. Data Cleaning & Normalization
    3. Exploratory Data Analysis (EDA)
    4. Train/Test Split
    5. Model Training (Logistic Regression, Random Forest, SVM, XGBoost)
    6. Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
    7. Feature Importance Analysis
    8. Export of all metrics/plots for the report
================================================================================
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
DATA_PATH = "data/diabetes.csv"
PLOTS_DIR = "plots"
RESULTS_PATH = "results.json"
RANDOM_STATE = 42
sns.set_theme(style="whitegrid", palette="muted")
os.makedirs(PLOTS_DIR, exist_ok=True)

COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

# ------------------------------------------------------------------------------
# 1. ETHICAL DATA HANDLING — logged explicitly as part of the pipeline
# ------------------------------------------------------------------------------
ETHICS_NOTES = """
ETHICAL & PRIVACY SAFEGUARDS APPLIED IN THIS PROJECT:
1. De-identified data only: the dataset carries no patient names, addresses,
   or identifiers of any kind (fully anonymized public research dataset).
2. Local processing: all analysis runs locally/in-session; no patient record
   is transmitted to a third party or external API.
3. Aggregate reporting: all reported outputs (metrics, plots) describe group-
   level model performance, never a single traceable patient's record.
4. Bias awareness: the dataset represents a specific population (Pima Indian
   heritage, female patients, age 21+), which limits generalizability. Any
   real deployment would need re-validation on the target population and a
   fairness audit across age, ethnicity, and other subgroups before clinical use.
5. No autonomous clinical action: this model is a decision-support / screening
   aid only. It must never replace a licensed clinician's diagnosis, and all
   predictions should be reviewed by a qualified medical professional.
6. Consent & governance: any real-world extension of this pipeline to hospital
   data would require informed patient consent, IRB/ethics-committee approval,
   and compliance with applicable health-data law (e.g., HIPAA in the US,
   India's DPDP Act 2023 for health data, or GDPR in the EU).
"""
print(ETHICS_NOTES)

# ------------------------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)
print(f"Loaded dataset: {df.shape[0]} records, {df.shape[1]} columns\n")

# ------------------------------------------------------------------------------
# 3. DATA CLEANING — biologically impossible zeros are treated as missing
# ------------------------------------------------------------------------------
zero_invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df_clean = df.copy()
for col in zero_invalid_cols:
    df_clean[col] = df_clean[col].replace(0, np.nan)

missing_summary = df_clean[zero_invalid_cols].isna().sum()
print("Missing values introduced (biologically-invalid zeros treated as NaN):")
print(missing_summary, "\n")

# Capture medians BEFORE imputation — reused later by the Streamlit app so that
# a single new patient record can be cleaned exactly the same way as training data.
impute_medians = {col: float(df_clean[col].median()) for col in zero_invalid_cols}

# Median imputation (robust to outliers, standard for clinical vitals)
for col in zero_invalid_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Remove exact duplicate patient rows if any
n_dupes = df_clean.duplicated().sum()
df_clean = df_clean.drop_duplicates()
print(f"Duplicate rows removed: {n_dupes}")
print(f"Final clean dataset shape: {df_clean.shape}\n")

df_clean.to_csv("data/diabetes_cleaned.csv", index=False)

# ------------------------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------------------------
# 4a. Class balance
plt.figure(figsize=(5, 4))
ax = sns.countplot(x="Outcome", data=df_clean, hue="Outcome", legend=False,
                    palette=["#4C72B0", "#C44E52"])
ax.set_xticks([0, 1])
ax.set_xticklabels(["No Diabetes (0)", "Diabetes (1)"])
plt.title("Class Distribution: Diabetes Outcome")
plt.ylabel("Number of Patients")
plt.xlabel("")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_class_distribution.png", dpi=150)
plt.close()

# 4b. Correlation heatmap
plt.figure(figsize=(8, 6))
corr = df_clean.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/02_correlation_heatmap.png", dpi=150)
plt.close()

# 4c. Feature distributions by outcome
fig, axes = plt.subplots(3, 3, figsize=(14, 11))
axes = axes.flatten()
for i, col in enumerate(COLUMNS[:-1]):
    sns.kdeplot(data=df_clean, x=col, hue="Outcome", fill=True, alpha=0.4, ax=axes[i],
                palette=["#4C72B0", "#C44E52"], common_norm=False)
    axes[i].set_title(col)
axes[-1].axis("off")
plt.suptitle("Feature Distributions by Diabetes Outcome", y=1.01, fontsize=14)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

print("EDA plots saved to /plots\n")

# ------------------------------------------------------------------------------
# 5. TRAIN/TEST SPLIT + NORMALIZATION
# ------------------------------------------------------------------------------
X = df_clean.drop(columns=["Outcome"])
y = df_clean["Outcome"]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train set: {X_train.shape[0]} patients | Test set: {X_test.shape[0]} patients\n")

# ------------------------------------------------------------------------------
# 6. MODEL TRAINING
# ------------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=6, random_state=RANDOM_STATE),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        eval_metric="logloss", random_state=RANDOM_STATE
    ),
}

results = {}
roc_data = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    use_scaled = name in ["Logistic Regression", "Support Vector Machine"]
    Xtr = X_train_scaled if use_scaled else X_train
    Xte = X_test_scaled if use_scaled else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring="accuracy")

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "cv_accuracy_mean": round(cv_scores.mean(), 4),
        "cv_accuracy_std": round(cv_scores.std(), 4),
    }
    results[name] = metrics
    roc_data[name] = roc_curve(y_test, y_proba)

    print(f"--- {name} ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print()

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Diabetes", "Diabetes"],
                yticklabels=["No Diabetes", "Diabetes"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"{PLOTS_DIR}/cm_{safe_name}.png", dpi=150)
    plt.close()

# ------------------------------------------------------------------------------
# 7. MODEL COMPARISON CHART
# ------------------------------------------------------------------------------
results_df = pd.DataFrame(results).T
results_df.to_csv("model_comparison_metrics.csv")

plt.figure(figsize=(9, 5))
metric_cols = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
results_df[metric_cols].plot(kind="bar", figsize=(10, 5), colormap="viridis")
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/04_model_comparison.png", dpi=150)
plt.close()

# ROC curves — all models
plt.figure(figsize=(6, 5))
for name, (fpr, tpr, _) in roc_data.items():
    auc_val = results[name]["roc_auc"]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — All Models")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/05_roc_curves.png", dpi=150)
plt.close()

# ------------------------------------------------------------------------------
# 8. FEATURE IMPORTANCE ANALYSIS
# ------------------------------------------------------------------------------
# Random Forest importance
rf_model = models["Random Forest"]
rf_importance = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)

# XGBoost importance
xgb_model = models["XGBoost"]
xgb_importance = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=False)

# Logistic Regression coefficients (standardized, so directly comparable)
lr_model = models["Logistic Regression"]
lr_importance = pd.Series(np.abs(lr_model.coef_[0]), index=feature_names).sort_values(ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
rf_importance.plot(kind="barh", ax=axes[0], color="#4C72B0")
axes[0].set_title("Random Forest — Feature Importance")
axes[0].invert_yaxis()

xgb_importance.plot(kind="barh", ax=axes[1], color="#55A868")
axes[1].set_title("XGBoost — Feature Importance")
axes[1].invert_yaxis()

lr_importance.plot(kind="barh", ax=axes[2], color="#C44E52")
axes[2].set_title("Logistic Regression — |Coefficient|")
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/06_feature_importance.png", dpi=150)
plt.close()

feature_importance_summary = pd.DataFrame({
    "RandomForest": rf_importance,
    "XGBoost": xgb_importance,
    "LogisticRegression_abs_coef": lr_importance
}).sort_values("RandomForest", ascending=False)
feature_importance_summary.to_csv("feature_importance.csv")

print("Feature Importance Summary (Random Forest ranking):")
print(feature_importance_summary, "\n")

# ------------------------------------------------------------------------------
# 9. BEST MODEL SUMMARY
# ------------------------------------------------------------------------------
best_model_name = results_df["roc_auc"].astype(float).idxmax()
print(f"BEST PERFORMING MODEL (by ROC-AUC): {best_model_name}")
print(results_df.loc[best_model_name])

summary = {
    "ethics_notes": ETHICS_NOTES.strip(),
    "dataset_shape_raw": list(df.shape),
    "dataset_shape_clean": list(df_clean.shape),
    "missing_values_imputed": {k: int(v) for k, v in missing_summary.to_dict().items()},
    "duplicates_removed": int(n_dupes),
    "train_size": int(X_train.shape[0]),
    "test_size": int(X_test.shape[0]),
    "model_results": results,
    "best_model": best_model_name,
    "top_5_features_rf": rf_importance.head(5).round(4).to_dict(),
}
with open(RESULTS_PATH, "w") as f:
    json.dump(summary, f, indent=2)

print("\nAll results, metrics, and plots have been saved.")
print("Pipeline complete.")

# ------------------------------------------------------------------------------
# 10. SAVE MODEL ARTIFACTS FOR DEPLOYMENT (used by the Streamlit app, app.py)
#     Additive only — does not change any analysis or output above.
# ------------------------------------------------------------------------------
import joblib

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

joblib.dump(models, os.path.join(MODELS_DIR, "trained_models.joblib"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

deployment_meta = {
    "feature_names": feature_names,
    "best_model": best_model_name,
    "scaled_models": ["Logistic Regression", "Support Vector Machine"],
    "zero_invalid_cols": zero_invalid_cols,
    "impute_medians": impute_medians,
}
with open(os.path.join(MODELS_DIR, "deployment_meta.json"), "w") as f:
    json.dump(deployment_meta, f, indent=2)

print(f"Model artifacts saved to /{MODELS_DIR} for Streamlit deployment (app.py).")
