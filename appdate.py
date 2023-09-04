import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import numpy as np
import pickle

# Load the saved dataset
dataset_path = 'C:/Users/Rohit Shelar/Excelr/PROJECTS/Project 2 - Oil Price Forecasting/dataset.csv'
df2 = pd.read_csv(dataset_path)

# Load the saved scaler
scaler_path = 'C:/Users/Rohit Shelar/Excelr/PROJECTS/Project 2 - Oil Price Forecasting/scaler.pkl'
with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the trained LSTM model
model = load_model('C:/Users/Rohit Shelar/Excelr/PROJECTS/Project 2 - Oil Price Forecasting/trained_lstm_model.h5')

st.title("Oil Price Forecasting App")

st.write("Enter a future date to predict its oil price:")

# Input field for the future date
input_date_str = st.date_input("Future Date")

if st.button("Predict"):
    # Format the input date to match the dataset's date format
    formatted_input_date = input_date_str.strftime('%Y-%m-%d')

    # Fetch historical prices up to the formatted input date
    historical_prices = df2[df2['date'] < formatted_input_date]['value']

    if len(historical_prices) < 10:
        st.write("Insufficient data for prediction.")
    else:
        # Take the most recent 10 prices for prediction
        recent_prices = historical_prices.tail(10).values

        # Scale and reshape the input data
        scaled_input_data = scaler.transform(recent_prices.reshape(-1, 1))
        X = scaled_input_data.reshape(1, 10, 1)

        # Make the prediction using the loaded LSTM model
        predicted_price_scaled = model.predict(X)
        predicted_price = scaler.inverse_transform(predicted_price_scaled)

        st.write(f"Predicted Oil Price for {input_date_str}: {predicted_price[0][0]}")