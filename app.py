#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Judul aplikasi
st.title("Price Prediction")

# Load model
try:
    model = joblib.load("XGboost_GridSearchCV.jodlib")
    st.success("✅ Model successfully loaded!")
except FileNotFoundError:
    st.error("⚠️ Model file 'XGboost_GridSearchCV.jodlib' not found. Please check the file.")
    model = None

# Input form untuk fitur numerik
bedrooms = st.number_input("Number of Bedrooms", min_value=0, step=1)
beds = st.number_input("Number of Beds", min_value=0, step=1)
host_listings_count = st.number_input("Host Listings Count", min_value=0, step=1)
zipcode = st.text_input("Zipcode (Enter as a number)", "98101")  # Default Zipcode

# Validasi input
if zipcode.isnumeric():
    zipcode = int(zipcode)

    # Menyiapkan input data
    input_data = np.array([[bedrooms, beds, host_listings_count, zipcode]])

    # Tombol untuk prediksi
    if st.button("Predict Price"):
        if model is not None:
            try:
                # Prediksi harga
                prediction = model.predict(input_data)
                st.write(f"🏡 **Predicted Price: ${prediction[0]:,.2f}**")
            except Exception as e:
                st.error(f"⚠️ Error in prediction: {e}")
        else:
            st.error("⚠️ Model is not loaded. Please check the file and try again.")
else:
    st.warning("⚠️ Zipcode must be a number!")






