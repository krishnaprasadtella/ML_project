import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVC

# Load the pre-trained SVC model
model = joblib.load("pipeline_model.pkl")

# Title and Description
st.title("Crop Recommendation")
st.write("""
    This application recommends the best crop to plant based on the following inputs:
    Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH Value, and Rainfall.
""")

# Input fields
nitrogen = st.number_input('Nitrogen', min_value=0, max_value=100, value=0)
phosphorus = st.number_input('Phosphorus', min_value=0, max_value=100, value=0)
potassium = st.number_input('Potassium', min_value=0, max_value=100, value=0)
temperature = st.number_input('Temperature', min_value=0.0, max_value=50.0, value=0.0)
humidity = st.number_input('Humidity', min_value=0.0, max_value=100.0, value=0.0)
ph_value = st.number_input('pH Value', min_value=0.0, max_value=14.0, value=0.0)
rainfall = st.number_input('Rainfall', min_value=0.0, max_value=300.0, value=0.0)

# Predict the crop
input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall]])

# Add a submit button
if st.button('Submit'):
    # Make predictions
    prediction = model.predict(input_data)
    st.write("Prediction:", prediction)
