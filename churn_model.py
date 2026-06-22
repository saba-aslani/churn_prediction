"""
churn_model.py
Churn prediction pipeline on Telco Customer Churn dataset.

Dataset: IBM Telco Customer Churn (7,043 customers, 26% churn rate)
Target: Predict which customers will churn (cancel subscription)
"""

import logging
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
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


# ── 1. Load & Clean ───────────────────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    log.info(f"Loading {path}")
    df = pd.read_csv(path)

    # TotalCharges has 11 blank rows (new customers with tenure=0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank_count = df["TotalCharges"].isna().sum()
    log.info(f"Dropping {blank_count} rows with missing TotalCharges (tenure=0 customers)")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    # Binary target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    churn_rate = df["Churn"].mean()
    log.info(f"Dataset: {df.shape} | Churn rate: {churn_rate:.1%}")
    return df


# ── 2. Feature Engineering ────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    # ── Binary encode Yes/No columns ─────────────────────────────────────────
    yes_no_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"
    ]
    for col in yes_no_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = (df[col] == "Yes").astype(int)

    # ── Service columns: encode No/No service → 0, Yes → 1 ───────────────────
    service_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in service_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0})

    # ── Ordinal encode Contract ───────────────────────────────────────────────
    df["Contract"] = df["Contract"].map({
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2,
    })

    # ── One-hot encode remaining categoricals ─────────────────────────────────
    df = pd.get_dummies(df, columns=["gender", "InternetService", "PaymentMethod"], drop_first=True)

    # ── Engineered features ───────────────────────────────────────────────────
    # Customers paying more per month relative to their total are newer/higher risk
    df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # Total services subscribed — more services = more embedded = lower churn
    df["total_services"] = (
        df["PhoneService"] + df["MultipleLines"] + df["OnlineSecurity"] +
        df["OnlineBackup"] + df["DeviceProtection"] + df["TechSupport"] +
        df["StreamingTV"] + df["StreamingMovies"]
    )

    # Monthly charge relative to number of services
    df["charge_per_service"] = df["MonthlyCharges"] / (df["total_services"] + 1)

    # Is customer a senior with no partner and no dependents? (higher churn risk)
    df["isolated_senior"] = ((df["SeniorCitizen"] == 1) &
                              (df["Partner"] == 0) &
                              (df["Dependents"] == 0)).astype(int)

    # Log tenure to capture diminishing returns of loyalty
    df["log_tenure"] = np.log1p(df["tenure"])

    feature_cols = [c for c in df.columns if c not in ["customerID", "Churn"]]
    log.info(f"Features engineered: {len(feature_cols)}")
    return df, feature_cols


# ── 3. Model Training & Comparison ───────────────────────────────────────────
def train_and_compare(X_train, y_train, class_weights: dict) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Logistic Regression needs scaling — wrap in Pipeline
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight=class_weights, max_iter=1000, random_state=42, C=0.1
        )),
    ])

    models = {
        "Logistic Regression (Baseline)": lr_pipeline,
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=class_weights.get(1, 1) / class_weights.get(0, 1),
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        ),
    }

    results = {}
    log.info("\n" + "=" * 55)
    log.info("5-Fold CV Results (ROC-AUC)")
    log.info("=" * 55)

    for name, model in models.items():
        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        results[name] = {
            "model": model,
            "cv_auc_mean": scores.mean(),
            "cv_auc_std": scores.std(),
        }
        log.info(f"{name:<40} AUC = {scores.mean():.3f} ± {scores.std():.3f}")

    return results


# ── 4. Final Evaluation ───────────────────────────────────────────────────────
def evaluate_best_model(results, X_train, X_test, y_train, y_test, feature_names):
    best_name = max(results, key=lambda k: results[k]["cv_auc_mean"])
    best_model = results[best_name]["model"]
    log.info(f"\nBest model: {best_name}")

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    log.info("\n" + "=" * 55)
    log.info("FINAL TEST SET EVALUATION")
    log.info("=" * 55)
    log.info(f"ROC-AUC: {auc:.3f}")
    log.info("\n" + classification_report(y_test, y_pred, target_names=["Active", "Churned"]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Model Evaluation: {best_name}  |  AUC = {auc:.3f}",
                 fontsize=13, fontweight="bold")

    RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[0], name=best_name)
    axes[0].plot([0, 1], [0, 1], "k--", label="Random Guess")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Active", "Churned"], ax=axes[1]
    )
    axes[1].set_title("Confusion Matrix")

    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=axes[2], name=best_name)
    axes[2].set_title("Precision-Recall Curve")

    plt.tight_layout()
    plt.savefig("model_evaluation_telco.png", dpi=150, bbox_inches="tight")
    log.info("Saved: model_evaluation_telco.png")
    plt.show()

    return best_model, best_name, auc


# ── 5. SHAP ───────────────────────────────────────────────────────────────────
def explain_with_shap(model, X_test: pd.DataFrame) -> None:
    # If model is a Pipeline, extract the actual model
    actual_model = model.named_steps["model"] if hasattr(model, "named_steps") else model
    X_explain = (
        pd.DataFrame(
            model.named_steps["scaler"].transform(X_test), columns=X_test.columns
        )
        if hasattr(model, "named_steps")
        else X_test
    )

    log.info("Calculating SHAP values...")
    explainer = shap.TreeExplainer(actual_model) if hasattr(actual_model, "feature_importances_") \
        else shap.LinearExplainer(actual_model, X_explain)

    shap_values = explainer.shap_values(X_explain)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("SHAP Explainability — What drives churn?", fontsize=13, fontweight="bold")

    plt.sca(axes[0])
    shap.summary_plot(sv, X_explain, show=False, plot_type="bar", max_display=15)
    axes[0].set_title("Global Feature Importance (SHAP)")

    plt.sca(axes[1])
    shap.summary_plot(sv, X_explain, show=False, max_display=15)
    axes[1].set_title("SHAP Values  |  Red = increases churn probability")

    plt.tight_layout()
    plt.savefig("shap_analysis_telco.png", dpi=150, bbox_inches="tight")
    log.info("Saved: shap_analysis_telco.png")
    plt.show()


# ── 6. Save ───────────────────────────────────────────────────────────────────
def save_model(model, feature_names: list, metadata: dict) -> None:
    joblib.dump(
        {"model": model, "feature_names": feature_names, "metadata": metadata},
        "churn_model_telco.pkl",
    )
    log.info("Saved: churn_model_telco.pkl")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("Telco Churn Prediction Pipeline — Starting")

    df = load_and_clean(CSV_PATH)
    df, feature_cols = engineer_features(df)

    X = df[feature_cols]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    log.info(f"Class weights: { {k: round(v, 3) for k, v in class_weights.items()} }")

    results = train_and_compare(X_train, y_train, class_weights)
    best_model, best_name, auc = evaluate_best_model(
        results, X_train, X_test, y_train, y_test, feature_cols
    )

    explain_with_shap(best_model, X_test)

    save_model(
        best_model,
        feature_cols,
        metadata={
            "dataset": "Telco Customer Churn",
            "best_model": best_name,
            "test_auc": round(auc, 4),
            "cv_auc": round(results[best_name]["cv_auc_mean"], 4),
            "n_features": len(feature_cols),
            "train_size": len(X_train),
            "churn_rate": round(float(y.mean()), 4),
        },
    )

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
