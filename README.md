# Telco Customer Churn Prediction Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red)](https://churnprediction-customer.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![AUC](https://img.shields.io/badge/ROC--AUC-0.851-brightgreen)]()

An end-to-end machine learning platform for predicting customer churn in the telecommunications industry. Built with production-grade practices including hyperparameter tuning, threshold optimization, SHAP explainability, and business impact quantification.

---

## Business Problem

Customer churn is one of the most costly challenges in the telecom industry. Acquiring a new customer costs 5–7× more than retaining an existing one. This platform helps retention teams:

- Identify high-risk customers **before** they cancel
- Prioritize retention campaigns by revenue at risk
- Understand **why** customers churn — not just who
- Quantify the **dollar value** of the prediction model

---

## Live Demo

🔗 [Churn Intelligence Platform →](https://churnprediction-customer.streamlit.app/)

**Dashboard pages:**
| Page | Description |
|------|-------------|
| 📊 Overview | KPIs, churn distribution, business impact |
| 🔍 Customer Explorer | Filter and identify at-risk customers |
| 🤖 Live Predictor | Real-time churn probability for any customer |
| 📈 Model Performance | AUC, SHAP, confusion matrix, score distribution |

---

## Results

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.851** |
| Recall (Churned) | **0.93** @ optimized threshold |
| Precision (Churned) | 0.43 |
| Optimal Threshold | 0.29 (F2-optimized) |
| Estimated Net ROI | **$84,220** per test cohort |

### Model Comparison (5-Fold CV AUC)

| Model | AUC | Std |
|-------|-----|-----|
| Logistic Regression | **0.851** | ±0.005 |
| XGBoost | 0.850 | ±0.004 |
| Random Forest | 0.847 | ±0.005 |

> Logistic Regression matched tree-based models — indicating largely linear relationships between features and churn. Simpler model preferred for production (faster inference, easier to audit).

---

## Key Findings

**Top churn drivers (SHAP):**
1. `MonthlyCharges` — highest paying customers on fiber churn most
2. `InternetService_Fiber optic` — fiber customers have 3× higher churn rate than DSL
3. `log_tenure` — churn risk drops sharply after 24 months
4. `Contract` — month-to-month customers churn at 42% vs 2.8% for two-year contracts
5. `total_services` — each additional service reduces churn probability ~8%

**Business insight:** The highest-value customers (fiber optic, high monthly charges, month-to-month contract) are simultaneously the highest churn risk. Retention strategy should prioritize contract upgrades and service bundling for this segment.

---

## Project Architecture

```
churn_prediction/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # IBM Telco dataset (7,043 customers)
│
├── churn_model.py                    # Week 1: Feature engineering + model comparison
├── churn_tuning_v2.py                      # Week 2: Hyperparameter tuning + threshold optimization
├── app_churn.py                         # Week 3: Production Streamlit dashboard
│
├── churn_model_tuned.pkl                   # Saved model artifact with metadata
├── requirements.txt
└── README.md
```

---

## ML Pipeline

### 1. Feature Engineering
Built 27 features from raw transactional data:

```python
# Behavioral features
df["charges_per_tenure"]      = df["MonthlyCharges"] / (df["tenure"] + 1)
df["total_services"]          = sum of all active services
df["charge_per_service"]      = df["MonthlyCharges"] / (df["total_services"] + 1)
df["isolated_senior"]         = senior with no partner and no dependents
df["log_tenure"]              = np.log1p(df["tenure"])   # captures diminishing loyalty effect
```

### 2. Model Selection
Compared three models via **Stratified 5-Fold Cross-Validation** with class weighting to handle the 74/26 class imbalance:

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = "roc_auc"   # accuracy excluded — misleading on imbalanced data
```

### 3. Hyperparameter Tuning
Used `RandomizedSearchCV` (30 iterations per model) — better coverage vs. GridSearch at a fraction of the compute cost.

### 4. Threshold Optimization
Default threshold of 0.5 is rarely optimal. Optimized using **F2-score** (recall-weighted) because the cost of missing a churner (~$1,200 lost revenue) exceeds the cost of a false alarm (~$50 retention offer).

```
Default  (0.50): Recall=0.79, Precision=0.50
Optimized(0.29): Recall=0.93, Precision=0.43  ← recommended
```

### 5. Explainability
SHAP (SHapley Additive exPlanations) used for both global feature importance and individual customer explanations — enabling actionable recommendations per customer.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data Processing | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualization | Plotly, Matplotlib |
| Dashboard | Streamlit |
| Model Persistence | Joblib |

---

## Setup & Run

```bash
# 1. Clone
git clone https://github.com/saba-aslani/churn-prediction
cd churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model
python churn_tuning_v2.py

# 4. Launch dashboard
streamlit run app_churn.py
```

---

## Dataset

**IBM Telco Customer Churn** — [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- 7,043 customers × 21 features
- Churn rate: 26.6%
- Features: demographics, account info, services subscribed, charges

---

## Author

**Saba Aslani** — Data Analyst & Data Engineer

[![GitHub](https://img.shields.io/badge/GitHub-saba--aslani-black)](https://github.com/saba-aslani)

---

## What I Learned

This project taught me that model performance is only half the story. The other half is:
- Defining the right metric for the business problem (F2 over accuracy)
- Communicating results in business terms ($84K ROI, not just AUC=0.851)
- Making models explainable to non-technical stakeholders (SHAP)
- Knowing when a simpler model is better than a complex one (LR ≈ XGBoost here)
