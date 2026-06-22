"""
app_churn.py
Streamlit dashboard for Telco Churn Prediction.
Run: streamlit run app_churn.py
"""

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 12px; color: white;
        text-align: center; margin: 5px;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; opacity: 0.85; margin-top: 4px; }
    .risk-high { background: linear-gradient(135deg, #f5576c, #f093fb); }
    .risk-medium { background: linear-gradient(135deg, #f7971e, #ffd200); color: #333; }
    .risk-low { background: linear-gradient(135deg, #43e97b, #38f9d7); color: #333; }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #4a4a4a;
        border-left: 4px solid #667eea; padding-left: 10px; margin: 15px 0 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Data & Model Loading ──────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
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
    return df, feature_cols


@st.cache_resource
def load_model():
    artifact = joblib.load("churn_model_tuned.pkl")
    return artifact["model"], artifact["threshold"], artifact["feature_names"]


@st.cache_data
def compute_shap_values(_model, X_sample):
    actual_model = _model.named_steps["model"] if hasattr(_model, "named_steps") else _model
    X_explain = (
        pd.DataFrame(
            _model.named_steps["scaler"].transform(X_sample),
            columns=X_sample.columns
        )
        if hasattr(_model, "named_steps")
        else X_sample
    )
    if hasattr(actual_model, "feature_importances_"):
        explainer = shap.TreeExplainer(actual_model)
    else:
        explainer = shap.LinearExplainer(actual_model, X_explain)
    sv = explainer.shap_values(X_explain)
    return sv[1] if isinstance(sv, list) else sv, X_explain


# ── Load everything ───────────────────────────────────────────────────────────
df, feature_cols = load_and_prepare()
model, threshold, _ = load_model()

X = df[feature_cols]
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_prob_all = model.predict_proba(X)[:, 1]
y_pred_all = (y_prob_all >= threshold).astype(int)
df["churn_probability"] = y_prob_all
df["predicted_churn"] = y_pred_all
df["risk_level"] = pd.cut(
    y_prob_all,
    bins=[0, 0.3, 0.6, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=60)
    st.title("Churn Intelligence")
    st.caption("Telco Customer Analytics Platform")
    st.divider()

    page = st.radio(
        "Navigation",
        ["📊 Overview", "🔍 Customer Explorer", "🤖 Live Predictor", "📈 Model Performance"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown(f"**Model:** Logistic Regression")
    st.markdown(f"**AUC:** 0.851")
    st.markdown(f"**Threshold:** {threshold:.2f}")
    st.markdown(f"**Dataset:** {len(df):,} customers")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Customer Churn Overview")
    st.caption("Business intelligence dashboard — real-time churn risk analysis")

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    total = len(df)
    high_risk = (df["risk_level"] == "High Risk").sum()
    medium_risk = (df["risk_level"] == "Medium Risk").sum()
    actual_churn = y.sum()
    churn_rate = y.mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{total:,}</div>
            <div class="metric-label">Total Customers</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card risk-high">
            <div class="metric-value">{high_risk:,}</div>
            <div class="metric-label">High Risk Customers</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card risk-medium">
            <div class="metric-value">{medium_risk:,}</div>
            <div class="metric-label">Medium Risk Customers</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card risk-low">
            <div class="metric-value">{churn_rate:.1%}</div>
            <div class="metric-label">Actual Churn Rate</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Charts Row 1 ──────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Churn Probability Distribution</div>',
                    unsafe_allow_html=True)
        fig = px.histogram(
            df, x="churn_probability", color="risk_level",
            color_discrete_map={"Low Risk": "#43e97b", "Medium Risk": "#f7971e",
                                "High Risk": "#f5576c"},
            nbins=40, opacity=0.8,
            labels={"churn_probability": "Churn Probability", "count": "# Customers"},
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                      annotation_text=f"Threshold={threshold:.2f}")
        fig.update_layout(height=320, margin=dict(t=20, b=20), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Risk Level Breakdown</div>',
                    unsafe_allow_html=True)
        risk_counts = df["risk_level"].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={"Low Risk": "#43e97b", "Medium Risk": "#f7971e",
                                "High Risk": "#f5576c"},
            hole=0.45,
        )
        fig.update_layout(height=320, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # ── Charts Row 2 ──────────────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Churn Rate by Contract Type</div>',
                    unsafe_allow_html=True)
        contract_map = {0: "Month-to-Month", 1: "One Year", 2: "Two Year"}
        df["Contract_label"] = df["Contract"].map(contract_map)
        contract_churn = df.groupby("Contract_label")["Churn"].mean().reset_index()
        contract_churn.columns = ["Contract", "Churn Rate"]
        fig = px.bar(
            contract_churn, x="Contract", y="Churn Rate",
            color="Churn Rate", color_continuous_scale="RdYlGn_r",
            text=contract_churn["Churn Rate"].apply(lambda x: f"{x:.1%}"),
        )
        fig.update_layout(height=300, margin=dict(t=20, b=20), coloraxis_showscale=False)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Monthly Charges vs Churn Probability</div>',
                    unsafe_allow_html=True)
        sample = df.sample(min(500, len(df)), random_state=42)
        fig = px.scatter(
            sample, x="MonthlyCharges", y="churn_probability",
            color="risk_level",
            color_discrete_map={"Low Risk": "#43e97b", "Medium Risk": "#f7971e",
                                "High Risk": "#f5576c"},
            opacity=0.6, size_max=6,
            labels={"churn_probability": "Churn Probability",
                    "MonthlyCharges": "Monthly Charges ($)"},
        )
        fig.add_hline(y=threshold, line_dash="dash", line_color="red")
        fig.update_layout(height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # ── Business Impact ───────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">💰 Business Impact Estimate</div>',
                unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    avg_revenue = 1200
    offer_cost = 50
    retention_rate = 0.30

    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= threshold).astype(int)
    TP = int(((y_pred_test == 1) & (y_test == 1)).sum())
    FP = int(((y_pred_test == 1) & (y_test == 0)).sum())
    FN = int(((y_pred_test == 0) & (y_test == 1)).sum())

    saved = TP * retention_rate * avg_revenue
    spend = (TP + FP) * offer_cost
    net = saved - spend

    with col_a:
        st.metric("Estimated Saved Revenue", f"${saved:,.0f}",
                  help="TP × 30% retention rate × $1,200 avg annual revenue")
    with col_b:
        st.metric("Campaign Spend", f"${spend:,.0f}",
                  help="(TP + FP) × $50 retention offer cost")
    with col_c:
        st.metric("Net Value of Model", f"${net:,.0f}",
                  delta=f"${net:,.0f} positive ROI")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: CUSTOMER EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Customer Explorer":
    st.title("🔍 Customer Risk Explorer")

    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.multiselect(
            "Risk Level", ["High Risk", "Medium Risk", "Low Risk"],
            default=["High Risk"]
        )
    with col2:
        min_prob = st.slider("Min Churn Probability", 0.0, 1.0, 0.5, 0.05)
    with col3:
        contract_filter = st.multiselect(
            "Contract Type",
            ["Month-to-Month", "One Year", "Two Year"],
            default=["Month-to-Month", "One Year", "Two Year"]
        )

    contract_reverse = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}
    contract_vals = [contract_reverse[c] for c in contract_filter]

    filtered = df[
        (df["risk_level"].isin(risk_filter)) &
        (df["churn_probability"] >= min_prob) &
        (df["Contract"].isin(contract_vals))
    ].copy()

    st.caption(f"Showing {len(filtered):,} customers matching filters")

    display_cols = ["churn_probability", "risk_level", "tenure",
                    "MonthlyCharges", "TotalCharges", "Contract", "total_services"]
    display = filtered[display_cols].copy()
    display["Contract"] = display["Contract"].map(
        {0: "Month-to-Month", 1: "One Year", 2: "Two Year"}
    )
    display["churn_probability"] = display["churn_probability"].apply(lambda x: f"{x:.1%}")
    display = display.sort_values("churn_probability", ascending=False)
    display.columns = ["Churn Prob", "Risk Level", "Tenure (months)",
                       "Monthly Charges", "Total Charges", "Contract", "# Services"]

    st.dataframe(display, use_container_width=True, height=400)

    # Revenue at risk
    revenue_at_risk = filtered["MonthlyCharges"].sum() * 12
    st.warning(f"⚠️ Annual revenue at risk from filtered customers: **${revenue_at_risk:,.0f}**")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Live Predictor":
    st.title("🤖 Real-Time Churn Predictor")
    st.caption("Enter customer details to get instant churn probability and recommendations")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Account Info**")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
        paperless = st.toggle("Paperless Billing", True)
        senior = st.toggle("Senior Citizen", False)

    with col2:
        st.markdown("**Services**")
        phone = st.toggle("Phone Service", True)
        multiple_lines = st.toggle("Multiple Lines", False)
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.toggle("Online Security", False)
        online_backup = st.toggle("Online Backup", False)

    with col3:
        st.markdown("**Additional Services**")
        device_protection = st.toggle("Device Protection", False)
        tech_support = st.toggle("Tech Support", False)
        streaming_tv = st.toggle("Streaming TV", False)
        streaming_movies = st.toggle("Streaming Movies", False)
        partner = st.toggle("Has Partner", True)
        dependents = st.toggle("Has Dependents", False)

    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        total_charges = monthly_charges * tenure

        total_svcs = sum([phone, multiple_lines, online_security, online_backup,
                         device_protection, tech_support, streaming_tv, streaming_movies])

        input_data = {
            "SeniorCitizen": int(senior),
            "Partner": int(partner),
            "Dependents": int(dependents),
            "tenure": tenure,
            "PhoneService": int(phone),
            "MultipleLines": int(multiple_lines),
            "OnlineSecurity": int(online_security),
            "OnlineBackup": int(online_backup),
            "DeviceProtection": int(device_protection),
            "TechSupport": int(tech_support),
            "StreamingTV": int(streaming_tv),
            "StreamingMovies": int(streaming_movies),
            "Contract": contract_map[contract],
            "PaperlessBilling": int(paperless),
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "charges_per_tenure": monthly_charges / (tenure + 1),
            "total_services": total_svcs,
            "charge_per_service": monthly_charges / (total_svcs + 1),
            "isolated_senior": int(senior and not partner and not dependents),
            "log_tenure": np.log1p(tenure),
            "gender_Male": 0,
            "InternetService_Fiber optic": int(internet == "Fiber optic"),
            "InternetService_No": int(internet == "No"),
            "PaymentMethod_Credit card (automatic)": 0,
            "PaymentMethod_Electronic check": 0,
            "PaymentMethod_Mailed check": 0,
        }

        # Align with training features
        input_df = pd.DataFrame([input_data])
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_cols]

        prob = model.predict_proba(input_df)[0][1]
        is_churn = prob >= threshold

        st.divider()
        r1, r2, r3 = st.columns(3)

        with r1:
            color = "#f5576c" if prob > 0.6 else "#f7971e" if prob > 0.3 else "#43e97b"
            st.markdown(f"""<div class="metric-card" style="background:{color}">
                <div class="metric-value">{prob:.1%}</div>
                <div class="metric-label">Churn Probability</div>
            </div>""", unsafe_allow_html=True)

        with r2:
            risk = "🔴 HIGH RISK" if prob > 0.6 else "🟡 MEDIUM RISK" if prob > 0.3 else "🟢 LOW RISK"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="font-size:1.4rem">{risk}</div>
                <div class="metric-label">Risk Classification</div>
            </div>""", unsafe_allow_html=True)

        with r3:
            revenue_risk = monthly_charges * 12
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">${revenue_risk:,}</div>
                <div class="metric-label">Annual Revenue at Risk</div>
            </div>""", unsafe_allow_html=True)

        # Recommendations
        st.divider()
        st.markdown("**💡 Retention Recommendations**")

        recs = []
        if contract == "Month-to-month":
            recs.append("📋 Offer discounted annual contract upgrade")
        if monthly_charges > 70:
            recs.append("💰 Review pricing — consider loyalty discount")
        if total_svcs < 3:
            recs.append("🔧 Bundle additional services at reduced rate")
        if tenure < 12:
            recs.append("🎁 Early loyalty reward program enrollment")
        if not online_security and not tech_support:
            recs.append("🛡️ Offer free trial of security/support services")
        if not recs:
            recs.append("✅ Customer appears stable — maintain regular engagement")

        for rec in recs:
            st.info(rec)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance & Explainability")

    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= threshold).astype(int)
    auc = roc_auc_score(y_test, y_prob_test)

    m1, m2, m3, m4 = st.columns(4)
    report = classification_report(y_test, y_pred_test, output_dict=True)

    m1.metric("ROC-AUC", f"{auc:.3f}")
    m2.metric("Precision (Churn)", f"{report['1']['precision']:.2f}")
    m3.metric("Recall (Churn)", f"{report['1']['recall']:.2f}")
    m4.metric("F1 (Churn)", f"{report['1']['f1-score']:.2f}")

    st.divider()

    # SHAP
    st.markdown('<div class="section-header">SHAP Feature Importance</div>',
                unsafe_allow_html=True)
    st.caption("Which features drive churn predictions most?")

    with st.spinner("Computing SHAP values..."):
        shap_sample = X_test.sample(min(200, len(X_test)), random_state=42)
        sv, X_explain = compute_shap_values(model, shap_sample)

    mean_shap = pd.DataFrame({
        "Feature": X_test.columns,
        "Mean |SHAP|": np.abs(sv).mean(axis=0)
    }).sort_values("Mean |SHAP|", ascending=True).tail(15)

    fig = px.bar(
        mean_shap, x="Mean |SHAP|", y="Feature",
        orientation="h",
        color="Mean |SHAP|",
        color_continuous_scale="Purples",
        title="Top 15 Features by Mean |SHAP| Value",
    )
    fig.update_layout(height=500, coloraxis_showscale=False, margin=dict(l=10))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown('<div class="section-header">Model Decision Boundary</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            x=y_prob_test, color=y_test.astype(str),
            color_discrete_map={"0": "#43e97b", "1": "#f5576c"},
            labels={"x": "Predicted Churn Probability", "color": "Actual"},
            title="Score Distribution by Actual Class",
            nbins=40, opacity=0.7, barmode="overlay",
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                      annotation_text=f"Threshold={threshold:.2f}")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        tenure_bins = pd.cut(X_test["tenure"], bins=[0, 12, 24, 48, 72], labels=["0-12m", "12-24m", "24-48m", "48-72m"])
        churn_by_tenure = pd.DataFrame({
            "tenure_group": tenure_bins,
            "churn_prob": y_prob_test
        }).groupby("tenure_group")["churn_prob"].mean().reset_index()

        fig = px.bar(
            churn_by_tenure, x="tenure_group", y="churn_prob",
            color="churn_prob", color_continuous_scale="RdYlGn_r",
            labels={"churn_prob": "Avg Churn Probability", "tenure_group": "Tenure Group"},
            title="Average Churn Risk by Tenure",
            text=churn_by_tenure["churn_prob"].apply(lambda x: f"{x:.1%}"),
        )
        fig.update_layout(height=350, coloraxis_showscale=False)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

