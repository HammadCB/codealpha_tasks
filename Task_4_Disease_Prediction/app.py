import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="AI HealthGuard | Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

# 2. Custom "Fancy" Styling
st.markdown("""
    <style>
    .main { background-color: #f0f4f8; }
    .stButton>button {
        background-color: #004a99;
        color: white;
        border-radius: 12px;
        height: 3.5em;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #003366; border: 1px solid #fff; }
    .prediction-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    h1 { color: #003366; font-family: 'Helvetica Neue', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# 3. App Header
st.title("🩺 AI HealthGuard: Diagnostic Intelligence")
st.write("### Enterprise-grade Diabetes Risk Assessment System")
st.write("---")

# 4. User Input Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🧬 Vitals")
    glu = st.slider('Glucose Level (mg/dL)', 0, 200, 115)
    bmi = st.slider('BMI (Body Mass Index)', 0.0, 60.0, 28.5)
    age = st.number_input('Patient Age', 21, 100, 35)

with col2:
    st.markdown("### 🩸 Clinical Data")
    bp = st.slider('Blood Pressure (mm Hg)', 0, 140, 72)
    ins = st.number_input('Insulin Level (mu U/ml)', 0, 900, 85)
    stk = st.number_input('Skin Thickness (mm)', 0, 100, 20)

with col3:
    st.markdown("### 📋 History")
    preg = st.number_input('Pregnancies', 0, 20, 2)
    dpf = st.number_input('Diabetes Pedigree Function', 0.0, 2.5, 0.47)

# 5. Prediction Engine
if st.button("GENERATE DIAGNOSTIC REPORT"):
    try:
        # Load the "Brain" and "Filter" we exported from Colab
        model = joblib.load('disease_model.joblib')
        scaler = joblib.load('disease_scaler.joblib')
        
        # Feature Engineering (Must match your Colab logic exactly)
        input_data = pd.DataFrame({
            'Pregnancies': [preg],
            'Glucose': [glu],
            'BloodPressure': [bp],
            'SkinThickness': [stk],
            'Insulin': [ins],
            'BMI': [bmi],
            'DiabetesPedigree': [dpf],
            'Age': [age],
            'Glucose_Age_Ratio': [glu / (age + 1)],
            'Health_Score': [(bmi * glu) / 100]
        })
        
        # Transformation and Prediction
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]
        
        st.write("---")
        
        # Result Visualization
        if prediction == 1:
            st.markdown(f"""
                <div class="prediction-box" style="background-color: #ffdce0; color: #a94442; border: 2px solid #ebccd1;">
                    ⚠️ ALERT: High Diabetes Risk Detected<br>
                    <span style="font-size: 18px;">Probability Score: {probability:.1%}</span>
                </div>
            """, unsafe_allow_html=True)
            st.warning("Recommendation: Patient should undergo clinical laboratory testing.")
        else:
            st.markdown(f"""
                <div class="prediction-box" style="background-color: #dff0d8; color: #3c763d; border: 2px solid #d6e9c6;">
                    ✅ SUCCESS: Low Diabetes Risk Detected<br>
                    <span style="font-size: 18px;">Probability Score: {probability:.1%}</span>
                </div>
            """, unsafe_allow_html=True)
            st.success("Recommendation: Maintain current healthy lifestyle and routine checkups.")

    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        st.info("Make sure 'disease_model.joblib' and 'disease_scaler.joblib' are in the same folder.")