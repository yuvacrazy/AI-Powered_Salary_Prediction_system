# frontend_app.py
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from fpdf import FPDF
import time
from datetime import datetime
import tempfile
import os

# ----------------------------
# CONFIG (use your backend URL + API KEY)
# ----------------------------
BACKEND_BASE = "https://ai-powered-salary-prediction-system.onrender.com"
BACKEND_BASE = BACKEND_BASE.rstrip("/")  # ensure no trailing slash
API_URL_PREDICT = f"{BACKEND_BASE}/predict"
API_URL_ANALYZE = f"{BACKEND_BASE}/analyze"
API_URL_EXPLAIN = f"{BACKEND_BASE}/explain"

# WARNING: for production, put this in Streamlit secrets or env vars.
BACKEND_URL = "https://ai-powered-salary-prediction-system.onrender.com/predict"
API_KEY = "34nCrCfhGjOZbtZAezzgHnxD7Gb_zVyk1x3HzisCKzQHcV5h"
HEADERS = {"x-api-key": API_KEY, "Content-Type": "application/json"}

response = requests.post(BACKEND_URL, json=data, headers=HEADERS)

st.set_page_config(page_title="SmartPay | AI Salary Intelligence", page_icon="💼", layout="wide")

# ----------------------------
# PDF REPORT GENERATOR
# ----------------------------
def generate_pdf_report(data, salary, kpis):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "SmartPay – AI Salary Intelligence Report", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(6)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Input Employee Details", ln=True)
    pdf.set_font("Arial", "", 11)
    for key, val in data.items():
        pdf.cell(0, 7, f"{key.capitalize().replace('_',' ')}: {val}", ln=True)
    pdf.ln(6)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Predicted Salary", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 7, f"Predicted Annual Salary (USD): ${salary:,.2f}", ln=True)
    pdf.ln(6)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Key Insights", ln=True)
    pdf.set_font("Arial", "", 11)
    for k, v in kpis.items():
        pdf.cell(0, 7, f"{k}: {v}", ln=True)

    pdf.ln(8)
    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(0, 7, "Developed by Yuvaraja P | Final Year CSE (IoT), Paavai Engineering College")

    temp_path = os.path.join(tempfile.gettempdir(), "SmartPay_Report.pdf")
    pdf.output(temp_path)
    return temp_path

# ----------------------------
# CSS / Theme (glassy corporate)
# ----------------------------
st.markdown(
    """
    <style>
    .glass-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(8px);
        border-radius: 14px;
        padding: 18px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .title {
        text-align:center;
        font-size:36px;
        font-weight:800;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {text-align:center; color:#c9d1d9; margin-bottom:20px;}
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<div class='title'>SmartPay – AI Salary Intelligence Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict · Analyze · Explain · Export Reports</div>", unsafe_allow_html=True)

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["💰 Prediction", "📊 Analysis", "🧠 Model Insights"])

# ----------------------------
# TAB 1: Prediction
# ----------------------------
with tab1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Input Employee Attributes")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=16, max_value=100, value=28)
        education = st.selectbox("Education", ["High School", "Bachelor’s", "Master’s", "PhD"])
    with col2:
        job_title = st.text_input("Job Title", "Software Engineer")
        hours_per_week = st.slider("Hours per Week", min_value=1, max_value=100, value=40)
    with col3:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    if st.button("🔍 Predict Salary", use_container_width=True):
     # Define backend URL and headers
BACKEND_URL = "https://ai-powered-salary-prediction-system.onrender.com/predict"
API_KEY = "34nCrCfhGjOZbtZAezzgHnxD7Gb_zVyk1x3HzisCKzQHcV5h"
HEADERS = {"x-api-key": API_KEY, "Content-Type": "application/json"}

# Send data to FastAPI backend
response = requests.post(BACKEND_URL, json=data, headers=HEADERS)

if response.status_code == 200:
    result = response.json()
    salary = float(result.get("predicted_salary_usd", 0))
    kpis = {"Prediction Confidence": "High", "Model Version": "LightGBM v1.0"}

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=salary,
        title={'text': "Predicted Annual Salary (USD)", 'font': {'size': 20, 'color': "#00c6ff"}},
        gauge={
            'axis': {'range': [0, 250000]},
            'bar': {'color': "#00c6ff"},
            'steps': [
                {'range': [0, 50000], 'color': '#1e1e1e'},
                {'range': [50000, 120000], 'color': '#24292f'},
                {'range': [120000, 250000], 'color': '#30363d'}
            ]
        }
    ))
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"<h3 style='text-align:center; color:#00c6ff;'>Predicted Salary: ${salary:,.2f}</h3>", unsafe_allow_html=True)

    pdf_path = generate_pdf_report(data, salary, kpis)
    with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="📄 Download Report (PDF)",
            data=pdf_file,
            file_name="SmartPay_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

elif response.status_code in [401, 403]:
    st.error("🔒 Authentication Error: Check your API key or backend security settings.")
else:
    st.error(f"⚠️ API Error {response.status_code}: {response.text}")

   
# ----------------------------
# TAB 2: Analysis
# ----------------------------
with tab2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Dataset Insights (from backend)")
    st.write("This fetches precomputed summary from backend `/analyze` endpoint.")
    if st.button("Fetch Analysis from Backend"):
        with st.spinner("Fetching analysis..."):
            try:
                r = requests.get(API_URL_ANALYZE, headers=HEADERS, timeout=20)
                if r.status_code == 200:
                    payload = r.json()
                    summary = payload.get("summary", {})
                    st.metric("Records", summary.get("record_count", "—"))
                    st.metric("Average Salary (USD)", f"${summary.get('average_salary', 0):,.2f}")
                    st.metric("Max Salary (USD)", f"${summary.get('max_salary', 0):,.2f}")
                    st.metric("Min Salary (USD)", f"${summary.get('min_salary', 0):,.2f}")
                elif r.status_code in (401,403):
                    st.error("Auth error for /analyze. Check API key.")
                else:
                    st.error(f"/analyze returned {r.status_code}: {r.text}")
            except Exception as e:
                st.error(f"Error fetching analysis: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.info("You can also upload a CSV locally to explore visualizations.")
    uploaded = st.file_uploader("Upload salary CSV (optional) — column: salary_in_usd", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if "salary_in_usd" in df.columns:
            fig = px.histogram(df, x="salary_in_usd", nbins=40, title="Salary Distribution")
            st.plotly_chart(fig, use_container_width=True)
            if "education" in df.columns:
                edu = df.groupby("education")["salary_in_usd"].mean().reset_index()
                st.bar_chart(edu.set_index("education")["salary_in_usd"])
        else:
            st.warning("Uploaded CSV doesn't contain 'salary_in_usd' column.")

# ----------------------------
# TAB 3: Model Insights
# ----------------------------
with tab3:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Model Explainability (top features)")
    st.write("This fetches feature importances from backend `/explain` endpoint.")
    if st.button("Fetch Model Insights"):
        with st.spinner("Fetching explainability..."):
            try:
                r = requests.get(API_URL_EXPLAIN, headers=HEADERS, timeout=20)
                if r.status_code == 200:
                    payload = r.json()
                    top = payload.get("top_features", [])
                    if top:
                        df_top = pd.DataFrame(top)
                        fig = px.bar(df_top, x="importance", y="feature", orientation="h", title="Feature Importance")
                        st.plotly_chart(fig, use_container_width=True)
                        st.table(df_top)
                    else:
                        st.info("No feature importance returned.")
                elif r.status_code in (401,403):
                    st.error("Auth error for /explain. Check API key.")
                else:
                    st.error(f"/explain returned {r.status_code}: {r.text}")
            except Exception as e:
                st.error(f"Error fetching explainability: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("""<hr><div style='text-align:center;color:#8b949e;'>Developed by <b>Yuvaraja P</b> | Final Year CSE (IoT), Paavai Engineering College — Powered by FastAPI · LightGBM · Streamlit</div>""", unsafe_allow_html=True)


