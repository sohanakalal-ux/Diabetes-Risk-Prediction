# 🩺 Diabetes Risk Prediction Using Machine Learning

### Healthcare Predictive Analytics | Data Analytics Internship — Codec Technologies

An end-to-end machine learning project that analyzes clinical measurements and predicts the likelihood of diabetes using multiple classification algorithms. The project includes data preprocessing, exploratory data analysis, model training and comparison, feature importance analysis, ethical considerations, and an interactive **Streamlit web application** for demonstrating predictions.

---

## 📌 Project Overview

Diabetes is a major chronic health condition that can lead to serious complications when not identified and managed appropriately. Machine learning can be used to analyze patterns within clinical datasets and estimate diabetes risk based on measurable health attributes.

This project was developed as part of my **Data Analytics Internship at CodTech Technologies** to demonstrate a complete machine learning workflow, from raw healthcare data preprocessing to model development, evaluation, interpretation, and deployment through a user-friendly web interface.

The project uses the **Pima Indians Diabetes Database**, a publicly available research dataset containing clinical measurements and diabetes outcome information.

The final system allows users to enter clinical measurements through a Streamlit interface and receive an ML-based diabetes risk prediction and estimated probability.

> ⚠️ **Important:** This project is an educational and machine learning demonstration. It is **not a medical diagnostic system** and must not be used for real clinical decision-making.

---

## 🎯 Project Objectives

The primary objectives of this project are to:

* Analyze a real-world healthcare dataset.
* Perform data preprocessing and cleaning.
* Explore relationships between clinical features and diabetes outcomes.
* Prepare the data for machine learning.
* Train and compare multiple classification algorithms.
* Evaluate models using appropriate performance metrics.
* Identify important features contributing to model predictions.
* Select the best-performing model based on evaluation results.
* Save trained models and preprocessing artifacts for reuse.
* Build an interactive Streamlit application around the existing ML pipeline.
* Demonstrate responsible and ethical handling of healthcare-related data.

---

## 📊 Dataset

### Pima Indians Diabetes Database

**Source:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)

The dataset contains **768 records** and **8 clinical input features**, along with a binary target variable indicating diabetes outcome.

### Input Features

| Feature                    | Description                                |
| -------------------------- | ------------------------------------------ |
| Pregnancies                | Number of pregnancies                      |
| Glucose                    | Plasma glucose concentration               |
| Blood Pressure             | Diastolic blood pressure                   |
| Skin Thickness             | Triceps skin fold thickness                |
| Insulin                    | 2-Hour serum insulin                       |
| BMI                        | Body Mass Index                            |
| Diabetes Pedigree Function | Diabetes-related hereditary risk indicator |
| Age                        | Age of the individual                      |

### Target Variable

**Outcome**

* `0` → No diabetes indicated in the dataset
* `1` → Diabetes indicated in the dataset

The dataset is de-identified and does not contain patient names, addresses, or direct personal identifiers.

---

## 🔄 Machine Learning Workflow

The project follows an end-to-end machine learning pipeline:

```text
Raw Dataset
     ↓
Data Loading
     ↓
Data Cleaning & Preprocessing
     ↓
Exploratory Data Analysis
     ↓
Feature Preparation
     ↓
Train/Test Split
     ↓
Feature Scaling
     ↓
Model Training
     ↓
Model Comparison
     ↓
Feature Importance Analysis
     ↓
Best Model Selection
     ↓
Model & Preprocessing Artifact Storage
     ↓
Streamlit Deployment
     ↓
Interactive Diabetes Risk Prediction
```

---

## 🧹 Data Preprocessing

The preprocessing stage prepares the clinical data for machine learning.

Key steps include:

* Loading the diabetes dataset using Pandas.
* Assigning meaningful column names.
* Inspecting the dataset structure.
* Checking data quality and missing/invalid values.
* Handling clinically unrealistic zero values where appropriate.
* Preparing cleaned data for analysis.
* Separating input features from the target variable.
* Splitting the data into training and testing sets.
* Applying feature scaling where required by the models.

The same preprocessing methodology is reused by the Streamlit application so that predictions are generated consistently with the trained ML pipeline.

---

## 🤖 Machine Learning Models

Four classification algorithms were implemented and compared:

### 1. Logistic Regression

A linear classification algorithm used as an interpretable baseline for binary classification.

### 2. Random Forest

An ensemble learning method that combines multiple decision trees to improve prediction performance and robustness.

### 3. Support Vector Machine (SVM)

A supervised learning algorithm that identifies an optimal decision boundary between classes.

### 4. XGBoost

A gradient-boosting algorithm that builds an ensemble of decision trees sequentially and is widely used for structured/tabular datasets.

---

## 🏆 Model Evaluation

The models were evaluated using classification performance metrics including:

* Accuracy
* ROC-AUC
* Confusion Matrix
* ROC Curves

The project also stores model comparison results in:

```text
model_comparison_metrics.csv
```

### Best Performing Model

Based on the project's evaluation results:

**XGBoost** was selected as the best-performing model.

Current project evaluation:

* **Accuracy:** ~76.6%
* **ROC-AUC:** ~0.824

ROC-AUC is particularly useful for evaluating how well a binary classification model distinguishes between the two outcome classes across different classification thresholds.

The model comparison results are retained in the repository so that the selection of the final model is supported by measurable evaluation rather than an arbitrary choice.

---

## 🧬 Feature Importance Analysis

Feature importance analysis is included to understand which clinical variables have the greatest influence on the trained model.

The project generates:

```text
feature_importance.csv
```

and visualizations within the:

```text
plots/
```

directory.

This analysis improves model interpretability and helps demonstrate which input variables contribute most strongly to the model's predictions.

---

## 🖥️ Streamlit Web Application

The project has been extended with an interactive **Streamlit application**.

The application provides a professional web-based interface where users can enter clinical measurements and obtain an ML-based risk prediction.

### Application Features

#### 🩺 Diabetes Risk Prediction

Users can enter:

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

The application then processes the input using the project's existing preprocessing pipeline and uses the selected best-performing model to generate a prediction.

### 📈 Risk Probability

Where supported by the model, the application displays an estimated probability/risk score alongside the prediction.

### 📊 Model Performance

The application provides model comparison information for:

* Logistic Regression
* Random Forest
* SVM
* XGBoost

### 🧬 Feature Importance

The Streamlit interface also presents feature importance information to make the model's behavior easier to understand.

### ℹ️ Project Information

The application contains information about:

* Project objectives
* Dataset
* Machine learning methodology
* Ethical considerations
* Responsible use of healthcare predictions

---

## 🗂️ Project Structure

```text
Diabetes-Risk-Prediction/
│
├── app.py
│   └── Streamlit web application
│
├── diabetes_risk_prediction.py
│   └── Main machine learning pipeline
│
├── requirements.txt
│   └── Python dependencies
│
├── data/
│   ├── diabetes.csv
│   └── diabetes_cleaned.csv
│
├── models/
│   ├── trained_models.joblib
│   ├── scaler.joblib
│   └── deployment_meta.json
│
├── plots/
│   ├── 01_class_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_feature_distributions.png
│   ├── 04_model_comparison.png
│   ├── 05_roc_curves.png
│   ├── 06_feature_importance.png
│   ├── cm_logistic_regression.png
│   ├── cm_random_forest.png
│   ├── cm_support_vector_machine.png
│   └── cm_xgboost.png
│
├── feature_importance.csv
├── model_comparison_metrics.csv
├── results.json
└── README.md
```

---

## 🛠️ Technologies Used

### Programming Language

* **Python**

### Data Analysis

* **Pandas**
* **NumPy**

### Machine Learning

* **Scikit-learn**
* **XGBoost**

### Data Visualization

* **Matplotlib**
* **Seaborn**

### Model Persistence

* **Joblib**

### Application Development

* **Streamlit**

### Development Environment

* **Visual Studio Code**

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Diabetes-Risk-Prediction.git
```

Navigate to the project directory:

```bash
cd Diabetes-Risk-Prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Streamlit Application

Launch the application using:

```bash
streamlit run app.py
```

After running the command, Streamlit will provide a local URL similar to:

```text
http://localhost:8501
```

Open the URL in a browser to access the interactive application.

---

## 🔁 Model Reusability

The application is designed to avoid unnecessary model retraining on every page load.

Previously trained models and preprocessing artifacts are stored using Joblib and reused by the Streamlit application.

This allows the application to:

* Load trained models efficiently.
* Reuse the fitted preprocessing/scaling process.
* Generate predictions without retraining the entire pipeline for every interaction.
* Keep the deployment workflow closer to a practical ML application.

---

## 🔐 Ethical & Privacy Considerations

Healthcare machine learning requires careful consideration of privacy, bias, interpretability, and responsible use.

### 1. De-identified Data

The dataset is a publicly available research dataset and does not contain patient names, addresses, or direct personal identifiers.

### 2. Local Processing

The demonstration application performs its processing locally/in-session and does not send entered patient information to an external AI or healthcare API.

### 3. No Autonomous Clinical Decision-Making

The application is intended only as an educational ML demonstration and potential decision-support concept.

It must **not** replace a qualified healthcare professional's diagnosis or treatment decisions.

### 4. Dataset Bias and Generalizability

The dataset represents a specific population and therefore may not generalize to every demographic or clinical population.

Before any real-world healthcare deployment, a model would require:

* External validation
* Population-specific validation
* Bias and fairness assessment
* Clinical evaluation
* Appropriate monitoring

### 5. Privacy and Governance

Any real-world implementation involving identifiable patient records would require appropriate consent, security controls, institutional approval, and compliance with applicable healthcare and data-protection regulations.

---

## ⚠️ Disclaimer

**This application is an educational machine learning demonstration and is not a medical diagnostic tool.**

The prediction and probability generated by this application should not be interpreted as a medical diagnosis, treatment recommendation, or substitute for professional medical advice.

For actual medical concerns, users should consult an appropriately qualified healthcare professional.

---

## 📚 Key Learning Outcomes

Through this project, I gained practical experience in:

* Healthcare dataset analysis
* Data cleaning and preprocessing
* Exploratory Data Analysis
* Feature engineering/preparation
* Classification algorithms
* Model comparison
* Accuracy and ROC-AUC evaluation
* Confusion matrix analysis
* ROC curve analysis
* Feature importance interpretation
* Model persistence using Joblib
* Building an interactive ML application using Streamlit
* Reusing trained models in a deployment workflow
* Responsible AI and ethical considerations in healthcare ML

---

## 💼 Internship Context

This project was developed as part of my **Data Analytics Internship at Codec Technologies**.

The project demonstrates the application of data analytics and machine learning concepts to a healthcare prediction problem, while also extending the analytical pipeline into an interactive application suitable for demonstration.

---

## 🚀 Future Improvements

Potential future improvements include:

* Testing on larger and more diverse datasets.
* External validation using independent healthcare datasets.
* More extensive hyperparameter tuning.
* Cross-validation and additional evaluation metrics.
* Improved model explainability using techniques such as SHAP.
* Fairness evaluation across relevant demographic groups.
* Model monitoring and drift detection.
* Secure deployment architecture for appropriately governed healthcare environments.
* Integration with authenticated healthcare workflows only after appropriate clinical and regulatory validation.

---

## 👩‍💻 Author

**Sohana Kalal**

B.Tech — Computer Science & Engineering (Artificial Intelligence & Machine Learning)

### Project Type

**Data Analytics / Machine Learning / Healthcare Predictive Analytics**

### Internship

**Data Analytics Internship — Codec Technologies**

---

## ⭐ Project Summary

This project demonstrates an end-to-end approach to building a healthcare machine learning solution:

**Data → Preprocessing → EDA → Multiple ML Models → Evaluation → Feature Importance → Best Model Selection → Model Persistence → Streamlit Application**

The goal is not simply to generate a prediction, but to demonstrate the complete workflow required to take a machine learning problem from **raw data to an interpretable and interactive application**, while recognizing the ethical and practical limitations of applying machine learning to healthcare.
