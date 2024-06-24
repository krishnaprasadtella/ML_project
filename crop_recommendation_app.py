import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVC
import os
import gdown


# Update the Google Drive file ID and model path
file_id ="/content/drive/MyDrive/Streamlit project New 1"
model_path = "E:\\streamlit\\pipeline_model.pkl"

# Download the model from Google Drive if not already downloaded
if not os.path.exists(model_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, model_path, quiet=False)

# Load the existing model
loaded_model = joblib.load(model_path)

# Title and Description
st.title("Crop Recommendation System")
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


