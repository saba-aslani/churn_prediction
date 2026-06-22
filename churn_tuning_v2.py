"""
churn_tuning_v2.py
Week 2: Hyperparameter tuning + threshold optimization.

Goals:
- Find optimal hyperparameters via RandomizedSearchCV
- Optimize classification threshold beyond default 0.5
- Business framing: quantify cost of each error type
"""

import logging
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CSV_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


# ── 1. Load & Feature Engineering (same as v1) ────────────────────────────────
def load_and_prepare(path: str):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    yes_no_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in yes_no_cols:
        df[col] = (df[col] == "Yes").astype(int)

    service_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in service_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0,
                               "No internet service": 0, "No phone service": 0})

    df["Contract"] = df["Contract"].map(
        {"Month-to-month": 0, "One year": 1, "Two year": 2}
    )
    df = pd.get_dummies(df, columns=["gender", "InternetService", "PaymentMethod"],
                        drop_first=True)

    df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    df["total_services"] = (
        df["PhoneService"] + df["MultipleLines"] + df["OnlineSecurity"] +
        df["OnlineBackup"] + df["DeviceProtection"] + df["TechSupport"] +
        df["StreamingTV"] + df["StreamingMovies"]
    )
    df["charge_per_service"] = df["MonthlyCharges"] / (df["total_services"] + 1)
    df["isolated_senior"] = (
        (df["SeniorCitizen"] == 1) & (df["Partner"] == 0) & (df["Dependents"] == 0)
    ).astype(int)
    df["log_tenure"] = np.log1p(df["tenure"])

    feature_cols = [c for c in df.columns if c not in ["customerID", "Churn"]]
    log.info(f"Dataset ready: {df.shape} | Churn rate: {df['Churn'].mean():.1%}")
    return df[feature_cols], df["Churn"], feature_cols


# ── 2. Hyperparameter Tuning ──────────────────────────────────────────────────
def tune_models(X_train, y_train, class_weights: dict) -> dict:
    """
    RandomizedSearchCV instead of GridSearchCV.
    Why: GridSearch on large param grids is computationally expensive.
    RandomizedSearch samples n_iter combinations — good balance of
    coverage vs speed. n_iter=30 covers most of the important space.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Logistic Regression ───────────────────────────────────────────────────
    log.info("Tuning Logistic Regression...")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight=class_weights, max_iter=2000, random_state=42
        )),
    ])
    lr_params = {
        "model__C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        "model__solver": ["lbfgs", "liblinear"],
        "model__penalty": ["l2"],
    }
    lr_search = RandomizedSearchCV(
        lr_pipe, lr_params, n_iter=14, cv=cv,
        scoring="roc_auc", n_jobs=-1, random_state=42
    )
    lr_search.fit(X_train, y_train)
    log.info(f"LR best AUC: {lr_search.best_score_:.3f} | params: {lr_search.best_params_}")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    log.info("Tuning XGBoost...")
    xgb_params = {
        "n_estimators": [200, 300, 500],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2],
    }
    xgb = XGBClassifier(
        scale_pos_weight=class_weights.get(1, 1) / class_weights.get(0, 1),
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    xgb_search = RandomizedSearchCV(
        xgb, xgb_params, n_iter=30, cv=cv,
        scoring="roc_auc", n_jobs=-1, random_state=42
    )
    xgb_search.fit(X_train, y_train)
    log.info(f"XGB best AUC: {xgb_search.best_score_:.3f} | params: {xgb_search.best_params_}")

    # ── Random Forest ─────────────────────────────────────────────────────────
    log.info("Tuning Random Forest...")
    rf_params = {
        "n_estimators": [200, 300, 500],
        "max_depth": [4, 6, 8, None],
        "min_samples_leaf": [5, 10, 20],
        "max_features": ["sqrt", "log2"],
        "min_samples_split": [2, 5, 10],
    }
    rf = RandomForestClassifier(
        class_weight=class_weights, random_state=42, n_jobs=-1
    )
    rf_search = RandomizedSearchCV(
        rf, rf_params, n_iter=30, cv=cv,
        scoring="roc_auc", n_jobs=-1, random_state=42
    )
    rf_search.fit(X_train, y_train)
    log.info(f"RF best AUC:  {rf_search.best_score_:.3f} | params: {rf_search.best_params_}")

    results = {
        "Logistic Regression": {"model": lr_search.best_estimator_,
                                 "cv_auc": lr_search.best_score_},
        "XGBoost": {"model": xgb_search.best_estimator_,
                    "cv_auc": xgb_search.best_score_},
        "Random Forest": {"model": rf_search.best_estimator_,
                          "cv_auc": rf_search.best_score_},
    }

    log.info("\n" + "=" * 55)
    log.info("Tuning Summary (CV AUC)")
    log.info("=" * 55)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["cv_auc"]):
        log.info(f"{name:<30} AUC = {r['cv_auc']:.3f}")

    return results


# ── 3. Threshold Optimization ─────────────────────────────────────────────────
def optimize_threshold(model, X_test, y_test) -> float:
    """
    Default threshold is 0.5 — but this is rarely optimal.

    Business context:
    - False Negative (miss a churner): customer leaves → lose ~$1,200/year revenue
    - False Positive (flag active customer): unnecessary retention offer → ~$50 cost

    At 0.5 threshold, recall on churned = 79% but precision = 50%.
    By lowering threshold, we catch more churners at cost of more false positives.
    By raising threshold, we improve precision but miss more real churners.

    We optimize for F2-score (weights recall 2x over precision)
    because missing a churner is more costly than a false alarm.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    # F2 score: beta=2 means recall is twice as important as precision
    beta = 2
    f2_scores = (1 + beta**2) * (precisions * recalls) / (
        (beta**2 * precisions) + recalls + 1e-8
    )

    # F1 score at each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    best_f2_idx = np.argmax(f2_scores[:-1])
    best_f1_idx = np.argmax(f1_scores[:-1])

    best_f2_threshold = thresholds[best_f2_idx]
    best_f1_threshold = thresholds[best_f1_idx]

    log.info(f"\nDefault threshold (0.50): captures most churners but low precision")
    log.info(f"Optimal F1  threshold: {best_f1_threshold:.2f}")
    log.info(f"Optimal F2  threshold: {best_f2_threshold:.2f}  ← recommended (recall-focused)")

    # ── Plot threshold analysis ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Threshold Optimization Analysis", fontsize=13, fontweight="bold")

    axes[0].plot(thresholds, precisions[:-1], label="Precision", color="blue")
    axes[0].plot(thresholds, recalls[:-1], label="Recall", color="orange")
    axes[0].plot(thresholds, f2_scores[:-1], label="F2 Score", color="green", linewidth=2)
    axes[0].axvline(best_f2_threshold, color="red", linestyle="--",
                    label=f"Best F2 threshold = {best_f2_threshold:.2f}")
    axes[0].axvline(0.5, color="gray", linestyle=":", label="Default = 0.50")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Precision / Recall / F2 vs Threshold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Compare default vs optimized confusion matrices
    y_pred_default = (y_prob >= 0.5).astype(int)
    y_pred_optimized = (y_prob >= best_f2_threshold).astype(int)

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_optimized,
        display_labels=["Active", "Churned"],
        ax=axes[1],
    )
    axes[1].set_title(f"Confusion Matrix @ threshold = {best_f2_threshold:.2f}")

    plt.tight_layout()
    plt.savefig("threshold_optimization.png", dpi=150, bbox_inches="tight")
    log.info("Saved: threshold_optimization.png")
    plt.show()

    # ── Side-by-side comparison ───────────────────────────────────────────────
    log.info("\n" + "=" * 55)
    log.info("DEFAULT threshold (0.50):")
    log.info("=" * 55)
    log.info("\n" + classification_report(
        y_test, y_pred_default, target_names=["Active", "Churned"]
    ))
    log.info("=" * 55)
    log.info(f"OPTIMIZED threshold ({best_f2_threshold:.2f}):")
    log.info("=" * 55)
    log.info("\n" + classification_report(
        y_test, y_pred_optimized, target_names=["Active", "Churned"]
    ))

    return best_f2_threshold


# ── 4. Business Impact Quantification ─────────────────────────────────────────
def quantify_business_impact(model, X_test, y_test, threshold: float) -> None:
    """
    Translate model performance into dollar value.
    This is what senior data scientists do that juniors don't.
    Numbers are illustrative — replace with real business values.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    TP = ((y_pred == 1) & (y_test == 1)).sum()  # Correctly flagged churners
    FP = ((y_pred == 1) & (y_test == 0)).sum()  # Active customers flagged as churners
    FN = ((y_pred == 0) & (y_test == 1)).sum()  # Missed churners
    TN = ((y_pred == 0) & (y_test == 0)).sum()  # Correctly identified active

    # Business assumptions (replace with real values)
    revenue_per_customer_year = 1200   # avg annual revenue per customer ($)
    retention_offer_cost = 50          # cost of retention offer per customer ($)
    retention_success_rate = 0.30      # % of flagged churners who stay due to offer

    saved_revenue = TP * retention_success_rate * revenue_per_customer_year
    retention_spend = (TP + FP) * retention_offer_cost
    lost_revenue = FN * revenue_per_customer_year
    net_value = saved_revenue - retention_spend

    log.info("\n" + "=" * 55)
    log.info("BUSINESS IMPACT ANALYSIS")
    log.info("=" * 55)
    log.info(f"True Positives  (churners caught):   {TP}")
    log.info(f"False Positives (wasted offers):     {FP}")
    log.info(f"False Negatives (missed churners):   {FN}")
    log.info(f"\nEstimated saved revenue:   ${saved_revenue:,.0f}")
    log.info(f"Retention campaign spend:  ${retention_spend:,.0f}")
    log.info(f"Net value of model:        ${net_value:,.0f}")
    log.info(f"Revenue at risk (FN):      ${lost_revenue:,.0f}")
    log.info("\n* Assumptions: $1,200 annual revenue/customer, "
             "$50 retention offer, 30% retention success rate")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("Churn Tuning Pipeline v2 — Starting")

    X, y, feature_cols = load_and_prepare(CSV_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    # Tune all models
    results = tune_models(X_train, y_train, class_weights)

    # Pick best
    best_name = max(results, key=lambda k: results[k]["cv_auc"])
    best_model = results[best_name]["model"]
    log.info(f"\nOverall best: {best_name} (AUC = {results[best_name]['cv_auc']:.3f})")

    # Optimize threshold
    best_threshold = optimize_threshold(best_model, X_test, y_test)

    # Business impact
    quantify_business_impact(best_model, X_test, y_test, best_threshold)

    # Save
    joblib.dump(
        {
            "model": best_model,
            "feature_names": feature_cols,
            "threshold": best_threshold,
            "metadata": {
                "best_model": best_name,
                "cv_auc": round(results[best_name]["cv_auc"], 4),
                "optimal_threshold": round(best_threshold, 3),
            },
        },
        "churn_model_tuned.pkl",
    )
    log.info("Saved: churn_model_tuned.pkl")
    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
