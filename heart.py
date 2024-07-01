#import library needed
import pandas as pd
import streamlit as st 
import pickle
import numpy as np

# Load the best model
with open('heart_model.pkl', 'rb') as f:
    model = pickle.load(f)

death = [0, 1]

st.title("Heart Failure Detection")
st.write("This app correctly detect heart failure based on data")

# Creating Sidebar for inputs
st.sidebar.title("Inputs")
age = st.sidebar.slider("Age", 40.0, 95.0, step=1.0)
anaemia = st.sidebar.slider("Anaemia: 1 for Yes, 0 for No", 0.0,1.0,step=1.0)
creatinine_phosphokinase = st.sidebar.slider("Level of the creatinine phosphokinase enzyme in the blood (mcg/L)", 30.0, 1202.0, 1000.0)
diabetes = st.sidebar.slider("Diabetes: 1 for Yes, 0 for No", 0.0, 1.0,step=1.0)
ejection_fraction = st.sidebar.slider("Ejection Fraction: Percentage of blood leaving the heart at each contraction (percentage)", 14.0, 65.0, 50.0)
high_blood_pressure = st.sidebar.slider("High Blood Pressure: 1 for Yes, 0 for No", 0.0,1.0,step=1.0)
platelets = st.sidebar.slider("Platelets: Platelets in the blood (kiloplatelets/mL)", 122000.0, 427000.0,200000.0)
serum_creatinine = st.sidebar.slider("Serum Creatinine: Level of serum creatinine in the blood (mg/dL)", 0.6, 2.1, 1.0)
serum_sodium = st.sidebar.slider("Serum Sodium: Level of serum sodium in the blood (mEq/L)", 125.0, 148.0,130.0)
sex = st.sidebar.slider("Sex: 0 for Woman, 1 for Man", 0.0,1.0,step=1.0)
smoking = st.sidebar.slider("Smoking: 1 for Yes, 0 for No", 0.0,1.0,step=1.0)
time = st.sidebar.slider("Follow-up period (days)", 4.0,285.0,step=1.0)
# Button to trigger prediction
if st.button("Predict"):
# Getting Prediction from model
    inp = np.array([age, ejection_fraction, serum_creatinine, time])
    inp = np.expand_dims(inp, axis=0)
    prediction = model.predict(inp)

# Show Results when the button is clicked
    result = death[np.argmax(prediction)]
    if result == 1:
        st.write("**Heart Failure will happen and cause death event**")
    elif result == 0:
        st.write("**Heart Failure will not happen and cause death event**")
    else:
        st.write("Unknown")

