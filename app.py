import streamlit as st
import pickle
import numpy as np
import pandas as pd

with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Stroke Prediction App")
st.markdown("Enter patient details to check the risk of stroke.")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

if st.button("Predict Stroke Risk"):
    input_data = pd.DataFrame([[age, hypertension, heart_disease, avg_glucose_level, bmi]],
                              columns=['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi'])
    
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]  

    st.subheader("🔍 Result")
    if prediction == 1:
        st.error(f"High Risk of Stroke (Probability: {prob:.2%})")
    else:
        st.success(f"Low Risk of Stroke (Probability: {prob:.2%})")
