#import library needed
import pandas as pd
import streamlit as st 
import pickle
import numpy as np

# Load the best model
with open('heart_model.pkl', 'rb') as f:
    model = pickle.load(f)

death = [1, 0]

st.title("Heart Failure Detection")
st.write("This app correctly detect heart failure based on data")

# Creating Sidebar for inputs
st.sidebar.title("Inputs")
age = st.sidebar.slider("Age", 40.0, 95.0, step=1.0)
anaemia = st.sidebar.number_input("Anaemia: 1 for Yes, 0 for No", 0.0,1.0,0.0)
creatinine_phosphokinase = st.sidebar.number_input("Level of the creatinine phosphokinase enzyme in the blood (mcg/L)", 30.0, 1202.0, 1000.0)
diabetes = st.sidebar.number_input("Diabetes: 1 for Yes, 0 for No", 0.0, 1.0,0.0)
ejection_fraction = st.sidebar.number_input("Ejection Fraction: Percentage of blood leaving the heart at each contraction (percentage)", 14.0, 65.0, 50.0)
high_blood_pressure = st.sidebar.number_input("High Blood Pressure: 1 for Yes, 0 for No", 0.0,1.0,0.0)
platelets = st.sidebar.number_input("Platelets: Platelets in the blood (kiloplatelets/mL)", 122000.0, 427000.0,200000.0)
serum_creatinine = st.sidebar.number_input("Serum Creatinine: Level of serum creatinine in the blood (mg/dL)", 0.6, 2.1, 1.0)
serum_sodium = st.sidebar.number_input("Serum Sodium: Level of serum sodium in the blood (mEq/L)", 125.0, 148.0,130.0)
sex = st.sidebar.number_input("Sex: 0 for Woman, 1 for Man", 0.0,1.0,0.0)
smoking = st.sidebar.number_input("Smoking: 1 for Yes, 0 for No", 0.0,1.0,0.0)
time = st.sidebar.number_input("Time: Follow-up period (days)", 4.0, 285.0,5.0)

# Button to trigger prediction
if st.button("Predict"):
# Getting Prediction from model
    inp = np.array([age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time])
    inp = np.expand_dims(inp, axis=0)
    prediction = model.predict(inp)

# Show Results when the button is clicked
    result = death[np.argmax(prediction)]
    if result == 1:
        st.write("**Heart Failure will happen and cause death event**")
    else:
        st.write("**Heart Failure will not happen and cause death event**")

