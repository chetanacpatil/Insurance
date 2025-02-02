import streamlit as st
import pandas as pd
import pickle
import sklearn
import numpy as np
import joblib

model = joblib.load("insurancenw.pkl")

st.title(" Health Insurance Cost Prediction")

age = st.number_input("Age", min_value=18, max_value=100, value=30)

AnyTransplant = st.selectbox("AnyTransplant",[0,1], format_func=lambda X: "Yes" if X == 1 else "No")
AnyChronicDeseases = st.selectbox("AnyChronicDeseases",[0,1], format_func=lambda X: "Yes" if X == 1 else "No")

Historyofcancerinfamily = st.selectbox("Historyofcancerinfamily",[0,1], format_func=lambda X: "Yes" if X == 1 else "No")
Numberofmajorsurgeries = st.number_input("Number of major surgeries", min_value=0, max_value=5, value=0)
Height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
Weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)

bmi = Weight/((Height / 100)**2)
st.write(f"Calculated BMI: {bmi:.2f}")
if st.button("Predict Premium"):
    features = np.array([[age,  AnyTransplant, AnyChronicDeseases,  Historyofcancerinfamily, Numberofmajorsurgeries, bmi  ]])
    prediction = model.predict(features)[0]
    st.success(f"Estimated Insurance Premium: {prediction:2f}")