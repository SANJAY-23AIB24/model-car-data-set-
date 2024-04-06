import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor as xgb
import joblib


# Load the trained model
model = joblib.load('your_model_file.joblib')

# Function to predict car selling price
def predict_price(year, present_price, kms_driven, fuel_type, seller_type, transmission, owner):
    # Prepare input data as a DataFrame
    data = pd.DataFrame({
        'Year': [year],
        'Present_Price': [present_price],
        'Kms_Driven': [kms_driven],
        'Fuel_Type': [fuel_type],
        'Seller_Type': [seller_type],
        'Transmission': [transmission],
        'Owner': [owner]
    })
    # Make prediction
    prediction = model.predict(data)
    return prediction[0]

# Streamlit UI
def main():
    st.title('Car Selling Price Prediction')
    
    # Input fields
    year = st.number_input('Year of Manufacture', min_value=1990, max_value=2024, step=1)
    present_price = st.number_input('Present Price (in lakhs)', min_value=0.0)
    kms_driven = st.number_input('Kilometers Driven', min_value=0)
    fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
    seller_type = st.selectbox('Seller Type', ['Dealer', 'Individual'])
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
    owner = st.number_input('Number of Previous Owners', min_value=0)

    # Predict button
    if st.button('Predict'):
        # Make prediction
        prediction = predict_price(year, present_price, kms_driven, fuel_type, seller_type, transmission, owner)
        st.success(f'Predicted Selling Price: ₹{prediction:.2f} lakhs')

if __name__ == '__main__':
    main()
