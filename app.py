import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

# Load the saved dataset
dataset_path = 'C:/Users/Rohit Shelar/Excelr/PROJECTS/Project 2 - Oil Price Forecasting/dataset.csv'
df2 = pd.read_csv(dataset_path)

# Load the saved scaler
scaler_path = 'C:/Users/Rohit Shelar/Excelr/PROJECTS/Project 2 - Oil Price Forecasting/scaler.pkl'
with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Create the GRU model architecture
model_gru = Sequential()
model_gru.add(GRU(units=50, activation='relu', input_shape=(10, 1)))
model_gru.add(Dense(units=1))
model_gru.compile(optimizer='adam', loss='mean_squared_error')

# Load the saved model weights
weights_path = 'C:/Users/Rohit Shelar/Excelr/PROJECTS/Project 2 - Oil Price Forecasting/gru_model_weights.h5'
model_gru.load_weights(weights_path)

# Save the model using Keras's model.save method
model_path = 'C:/Users/Rohit Shelar/Excelr/PROJECTS/Project 2 - Oil Price Forecasting/trained_model.h5'

def predict_price(data):
    scaled_data = scaler.transform(data)
    sequences = []
    for i in range(len(scaled_data) - 10 + 1):
        sequences.append(scaled_data[i:i+10, 0])
    X = np.array(sequences)
    predictions = model_gru.predict(X)
    predictions = scaler.inverse_transform(predictions)
    return predictions

st.title("Oil Price Forecasting App")

st.write("Enter the last 10 days' oil price values:")

input_data = []
for i in range(10):
    value = st.number_input(f"Day {i+1}", value=0.0)
    input_data.append(value)

if st.button("Predict"):
    input_data = np.array(input_data).reshape(1, 10, 1)  # Reshape input data
    predictions = predict_price(input_data)
    st.write("Predicted Oil Prices for the next day:")
    st.write(predictions)
