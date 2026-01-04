import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image

# ==============================
# A. CONFIGURATION AND STYLING 
# ==============================
st.set_page_config(
    page_title="Credit Risk Scorecard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown("""
    <style>
    .stApp { background-color: #F7F9F9; color: #1F2E3A; }
    .stButton>button { background-color: #2E86C1; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- Global Definitions ---
MODEL_FILE = 'credit_scoring_model.joblib'
SCALER_FILE = 'feature_scaler.joblib'
FEATURE_NAMES = ['rev_util', 'age', 'late_30_59', 'debt_ratio', 'monthly_inc', 'open_credit', 'late_90', 'real_estate', 'late_60_89', 'dependents']


# ====================
# B. CORE FUNCTIONS 
# ====================

@st.cache_resource
def load_assets():
    """Loads the trained model and scaler once for efficient use."""
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: Model or Scaler file not found. Please run all preceding ML cells.")
        return None, None

def predict_score(model, scaler, features):
    """Predicts the probability of default and calculates a credit score."""
    try:
        input_df = pd.DataFrame([features], columns=FEATURE_NAMES)

        #  Apply the same scaling as training data
        input_scaled = scaler.transform(input_df)

        # 1. Predict Probability of Default (PoD)
        prob_default = model.predict_proba(input_scaled)[:, 1][0]

        # 2. Calculate Credit Score (Transformation)
        # Standard scorecard logic: Score = Base - (PoD * ScalingFactor)
        credit_score = int(850 - (prob_default * 500))
        if credit_score < 300: credit_score = 300 # Set floor

        return prob_default, credit_score
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return None, None


# ===========================
# C. MAIN APPLICATION LAYOUT 
# ===========================

st.title(":lock: Professional Credit Risk Scorecard")
st.subheader("Interactive Model Validation and Prediction Dashboard")

model, scaler = load_assets()


with st.sidebar:
    st.header("Model Performance")
    try:
        roc_image = Image.open('roc_curve_plot.png')

        st.image(roc_image, caption=f"AUC-ROC: **0.8621** (High-Accuracy Score)", use_column_width=True)
    except FileNotFoundError:
        st.warning("ROC Curve plot not found. Run Cell 15 first.")

    st.write("---")
    st.info("""
    **Architecture:** Random Forest Classifier\n
    **Evaluation Metric:** AUC-ROC\n
    **Model Status:** Ready for Prediction
    """)

# --- MAIN INPUT SECTION ---

if model is not None and scaler is not None:
    st.header("Applicant Profile Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Financial Usage")
        rev_util = st.number_input("Revolving Utilization Rate", min_value=0.0, value=0.5, format="%.4f")
        debt_ratio = st.number_input("Debt Ratio", min_value=0.0, value=0.3, format="%.4f")
        monthly_inc = st.number_input("Monthly Income ($)", min_value=0.0, value=5000.0, step=100.0)

    with col2:
        st.subheader("Demographics")
        age = st.number_input("Age", min_value=18, max_value=100, value=45)
        dependents = st.number_input("Number of Dependents", min_value=0.0, value=1.0, step=1.0)
        open_credit = st.number_input("Open Credit Lines", min_value=0.0, value=10.0, step=1.0)

    with col3:
        st.subheader("Delinquency Events")
        late_30_59 = st.number_input("Late 30-59 Days", min_value=0.0, value=0.0, step=1.0)
        late_60_89 = st.number_input("Late 60-89 Days", min_value=0.0, value=0.0, step=1.0)
        late_90 = st.number_input("Late 90+ Days", min_value=0.0, value=0.0, step=1.0)
        real_estate = st.selectbox("Real Estate Loans (e.g., Mortgage)", [0.0, 1.0, 2.0], index=1)

    st.markdown("---")

# --- PREDICTION BUTTON AND OUTPUT ---
if st.button("Generate Credit Score", use_container_width=True):
    input_data = [rev_util, age, late_30_59, debt_ratio, monthly_inc, open_credit, late_90, real_estate, late_60_89, dependents]

    # Input Validation 
    if all(x is not None and x >= 0 for x in input_data):
        prob_default, credit_score = predict_score(model, scaler, input_data)

        if prob_default is not None:
            st.subheader("Prediction Results:")
            score_col, default_col = st.columns(2)

            with score_col:
                st.metric(label="Generated Credit Score", value=credit_score, delta="Predicted")

            with default_col:
                # Use delta_color="inverse" so that a higher PoD shows as a "negative" (Red) change
                st.metric(label="Probability of Default (PoD)", value=f"{prob_default:.2%}", delta="Risk Level", delta_color="inverse")

            # --- DYNAMIC VISUAL FEEDBACK FOR VIDEO ---
            if prob_default < 0.2:
                risk_message = "LOW RISK: Applicant is predicted to be highly creditworthy."
                st.success(f"Final Assessment: {risk_message}")
            else:
                risk_message = "HIGH RISK: Proceed with caution, as probability of default is high."
                st.error(f"Final Assessment: {risk_message}")
    else:
        st.error("Input Error: Please ensure all input values are valid (non-negative).")