import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data_generator import generate_dataset
from model_utils import train_model, evaluate_model, get_model_bundle

st.set_page_config(page_title="LoanLens | Approval Intelligence", page_icon="💳", layout="wide")

# -------------------- DESIGN SYSTEM --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f5f7fb; }
[data-testid="stSidebar"] { background: #101828; }
[data-testid="stSidebar"] * { color: #e5eaf2 !important; }
[data-testid="stSidebar"] .stRadio label { padding: 8px 10px; border-radius: 10px; }
h1,h2,h3 { font-family: 'Manrope', sans-serif; color: #101828; }
.hero { background: linear-gradient(120deg,#111c3a 0%,#173b71 58%,#2563eb 100%); padding: 28px 32px; border-radius: 22px; color: white; margin-bottom: 22px; }
.hero h1 { color: white; margin: 0; font-size: 2.05rem; }
.hero p { color: #cbd8f3; margin: 8px 0 0 0; }
.metric-card { background: white; border: 1px solid #e6eaf0; border-radius: 16px; padding: 18px 20px; box-shadow: 0 3px 12px rgba(16,24,40,.04); }
.metric-label { color: #667085; font-size: 0.83rem; font-weight: 600; }
.metric-value { color: #101828; font-size: 1.65rem; font-weight: 800; margin-top: 5px; font-family: 'Manrope', sans-serif; }
.metric-note { color: #667085; font-size: .78rem; margin-top: 4px; }
.section-card { background: white; border: 1px solid #e6eaf0; border-radius: 16px; padding: 20px; }
.small-muted { color: #667085; font-size: .88rem; }
div.stButton > button { border-radius: 10px; font-weight: 700; }
div[data-testid="stForm"] { background: white; border: 1px solid #e6eaf0; padding: 22px; border-radius: 18px; }
div[data-testid="stMetric"] { background: white; border: 1px solid #e6eaf0; padding: 14px; border-radius: 14px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return generate_dataset(1200, seed=42)

@st.cache_resource
def load_bundle():
    model_path = os.path.join(ROOT, "models", "loan_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    bundle = train_model(load_data())
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    return bundle

data = load_data()
bundle = load_bundle()
model = bundle["model"]
metrics = bundle["metrics"]

# Sidebar
with st.sidebar:
    st.markdown("## 💳 LoanLens")
    st.caption("APPROVAL INTELLIGENCE")
    st.markdown("---")
    page = st.radio("NAVIGATION", ["Overview", "Loan Prediction", "Data Analytics", "Model Performance"], label_visibility="visible")
    st.markdown("---")
    st.markdown("**Model status**")
    st.success("ML model ready")
    st.caption("Random Forest • Demo dataset")
    st.markdown("---")
    st.caption("Portfolio project • Built with Python + Streamlit")

def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def metric_card(label, value, note=""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

# Overview
if page == "Overview":
    hero("Loan Approval Intelligence", "Understand application patterns and explore machine-learning-powered approval predictions.")
    st.markdown("### Portfolio snapshot")
    cols = st.columns(4)
    approval_rate = data["loan_status"].mean() * 100
    with cols[0]: metric_card("Applications", f"{len(data):,}", "Demo dataset")
    with cols[1]: metric_card("Approval rate", f"{approval_rate:.1f}%", "Historical target distribution")
    with cols[2]: metric_card("Model accuracy", f"{metrics['accuracy']:.1%}", "Test-set evaluation")
    with cols[3]: metric_card("ROC-AUC", f"{metrics['roc_auc']:.3f}", "Model discrimination")
    st.markdown("")
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Approval distribution")
        fig = px.pie(data, names="loan_status", hole=.62, color="loan_status",
                      color_discrete_map={0:"#f59e0b",1:"#2563eb"})
        fig.update_traces(textinfo="percent+label", labels=["Rejected","Approved"])
        fig.update_layout(height=330, margin=dict(t=10,b=10,l=10,r=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Applicant income distribution")
        fig = px.histogram(data, x="applicant_income", color="loan_status", nbins=28,
                           color_discrete_map={0:"#f59e0b",1:"#2563eb"},
                           labels={"loan_status":"Approval","applicant_income":"Applicant income"})
        fig.update_layout(height=330, margin=dict(t=10,b=10,l=10,r=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.info("This dashboard uses a generated demonstration dataset. It is not connected to a bank or real applicant records.")

# Prediction
elif page == "Loan Prediction":
    hero("Loan Eligibility Predictor", "Enter applicant details to estimate the model's approval probability.")
    st.markdown("#### Applicant profile")
    with st.form("prediction_form"):
        a, b, c = st.columns(3)
        with a:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            married = st.selectbox("Marital status", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        with b:
            self_employed = st.selectbox("Self employed", ["No", "Yes"])
            applicant_income = st.number_input("Applicant monthly income (₹)", min_value=0, value=45000, step=5000)
            coapplicant_income = st.number_input("Co-applicant monthly income (₹)", min_value=0, value=0, step=5000)
            loan_amount = st.number_input("Loan amount (₹ thousands)", min_value=1, value=150, step=10)
        with c:
            loan_term = st.selectbox("Loan term (months)", [120, 180, 240, 300, 360], index=4)
            credit_history = st.selectbox("Credit history", ["Good / available", "Poor / unavailable"])
            property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
        submitted = st.form_submit_button("✨ Predict approval", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "gender": gender, "married": married, "dependents": dependents,
            "education": education, "self_employed": self_employed,
            "applicant_income": applicant_income, "coapplicant_income": coapplicant_income,
            "loan_amount": loan_amount, "loan_term": loan_term,
            "credit_history": 1 if credit_history == "Good / available" else 0,
            "property_area": property_area
        }])
        probability = float(model.predict_proba(row)[0][1])
        predicted = int(probability >= 0.5)
        st.markdown("---")
        left, right = st.columns([1, 1.2])
        with left:
            if predicted:
                st.success("### Predicted: APPROVAL LIKELY")
                st.markdown(f"Estimated approval probability: **{probability:.1%}**")
            else:
                st.warning("### Predicted: APPROVAL UNLIKELY")
                st.markdown(f"Estimated approval probability: **{probability:.1%}**")
            st.caption("This is a model estimate, not a financial decision or guarantee.")
        with right:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=probability*100,
                number={"suffix":"%","font":{"size":32}},
                gauge={"axis":{"range":[0,100]}, "bar":{"color":"#2563eb"},
                       "steps":[{"range":[0,50],"color":"#fee2e2"},{"range":[50,100],"color":"#dbeafe"}],
                       "threshold":{"line":{"color":"#101828","width":3},"thickness":0.8,"value":50}}))
            fig.update_layout(height=250, margin=dict(t=20,b=0,l=20,r=20))
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("View submitted applicant details"):
            st.dataframe(row, use_container_width=True, hide_index=True)

# Analytics
elif page == "Data Analytics":
    hero("Data Analytics", "Explore applicant demographics, income, loan amounts, and approval patterns.")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Loan amount by property area")
        fig = px.box(data, x="property_area", y="loan_amount", color="property_area",
                     color_discrete_sequence=["#2563eb","#14b8a6","#f59e0b"])
        fig.update_layout(height=350, margin=dict(t=10,b=10,l=10,r=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Approval rate by education")
        grouped = data.groupby("education", as_index=False)["loan_status"].mean()
        grouped["loan_status"] *= 100
        fig = px.bar(grouped, x="education", y="loan_status", text_auto=".1f",
                     color="education", color_discrete_sequence=["#2563eb","#14b8a6"])
        fig.update_layout(height=350, margin=dict(t=10,b=10,l=10,r=10), showlegend=False,
                         yaxis_title="Approval rate (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    c3,c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Applicant vs co-applicant income")
        fig = px.scatter(data, x="applicant_income", y="coapplicant_income", color="loan_status",
                         color_discrete_map={0:"#f59e0b",1:"#2563eb"}, opacity=.65)
        fig.update_layout(height=350, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Approval by credit history")
        grouped = data.groupby("credit_history", as_index=False)["loan_status"].mean()
        grouped["credit_history"] = grouped["credit_history"].map({1:"Good / available",0:"Poor / unavailable"})
        grouped["loan_status"] *= 100
        fig = px.bar(grouped, x="credit_history", y="loan_status", text_auto=".1f",
                     color="credit_history", color_discrete_sequence=["#2563eb","#f59e0b"])
        fig.update_layout(height=350, margin=dict(t=10,b=10,l=10,r=10), showlegend=False,
                         yaxis_title="Approval rate (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("#### Dataset preview")
    st.dataframe(data.head(100), use_container_width=True, hide_index=True)

# Performance
elif page == "Model Performance":
    hero("Model Performance", "Test-set metrics and diagnostic visualizations.")
    cols = st.columns(4)
    with cols[0]: metric_card("Accuracy", f"{metrics['accuracy']:.1%}")
    with cols[1]: metric_card("Precision", f"{metrics['precision']:.1%}")
    with cols[2]: metric_card("Recall", f"{metrics['recall']:.1%}")
    with cols[3]: metric_card("F1-score", f"{metrics['f1']:.3f}")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Confusion matrix")
        cm = metrics["confusion_matrix"]
        fig = px.imshow(cm, text_auto=True, x=["Predicted Rejected","Predicted Approved"],
                        y=["Actual Rejected","Actual Approved"], color_continuous_scale="Blues")
        fig.update_layout(height=350, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Feature importance")
        importance = pd.DataFrame({"feature":bundle["feature_names"],"importance":model.named_steps["classifier"].feature_importances_})
        importance = importance.sort_values("importance", ascending=True).tail(12)
        fig = px.bar(importance, x="importance", y="feature", orientation="h",
                     color="importance", color_continuous_scale="Blues")
        fig.update_layout(height=350, margin=dict(t=10,b=10,l=10,r=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("### Classification report")
    report = pd.DataFrame(metrics["classification_report"]).T
    st.dataframe(report.round(3), use_container_width=True)
    st.caption("Metrics are based on a held-out split of the generated demo dataset. They should not be interpreted as real-world lending performance.")
