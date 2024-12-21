
# Paste your final Streamlit integration code here.
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import sys
import joblib

# Global exception handler
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    st.error(f"An unexpected error occurred: {exc_value}")
    # Optionally log the exception details for debugging (not shown to the user)
    with open("error_log.txt", "a") as f:
        f.write(f"{datetime.now()} - {exc_type.__name__}: {exc_value}\n")

sys.excepthook = handle_exception


# Global multiplier for prices
price_multiplier = 1.0

# Load the trained model
try:
    model_filename = "house_price_model.pkl"
    with open(model_filename, 'rb') as file:
        model = joblib.load(file)

    # Check if the loaded model has the necessary attributes
    if not hasattr(model, 'predict'):
        raise AttributeError("Loaded model does not support predictions.")
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'house_price_model.pkl' exists.")
    st.stop()  # Stop the execution if the model is not available
except AttributeError as ae:
    st.error(f"Model compatibility issue: {ae}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading the model: {e}")
    st.stop()


# File to store the last update date
update_file = "last_update.txt"

# Function to check and perform yearly price update
# Function to check and perform yearly price update
def check_and_update_prices():
    try:
        # Load the last update date from the file
        with open(update_file, 'r') as file:
            last_update = datetime.strptime(file.read().strip(), "%Y-%m-%d")
    except FileNotFoundError:
        # If file doesn't exist, assume no prior updates
        last_update = datetime.min

    # Get the current date
    current_date = datetime.now()

    # Check if a year has passed
    if current_date.year > last_update.year:
        # Apply a 5% increase to prices using an external multiplier
        global price_multiplier
        price_multiplier *= 1.05

        # Save the new update date
        with open(update_file, 'w') as file:
            file.write(current_date.strftime("%Y-%m-%d"))

        return f"Prices updated for the year {current_date.year}."
    else:
        return "Prices are already up-to-date."


# Perform automatic price updates
update_message = check_and_update_prices()

# Set up the Streamlit app
st.title("House Price Prediction App")

# Display the update message
st.info(update_message)

# Input fields for user input
st.header("Enter Property Details:")
area = st.number_input("Area (in square feet):", min_value=100, max_value=10000, step=10)
try:
    # Validate the area input
    if area <= 0:
        raise ValueError("Area must be a positive number.")
except ValueError as e:
    st.error(f"Invalid input: {e}")

air_conditioning = st.selectbox("Air Conditioning:", ["Yes", "No"])
parking = st.selectbox("Parking:", ["Yes", "No"])

# Transform categorical inputs
try:
    air_conditioning_flag = 1 if air_conditioning.lower() == "yes" else 0
    parking_flag = 1 if parking.lower() == "yes" else 0

except Exception as e:
    st.error(f"An error occurred while processing the inputs: {e}")



if st.button("Predict Price"):
    try:
        # Prepare features for prediction
        features = np.array([[area, air_conditioning_flag, parking_flag]])

        # Ensure the feature dimensions match the model's expectations
        if features.shape[1] != model.n_features_in_:
            raise ValueError("Mismatch in input features for the prediction model.")

        # Predict and apply the multiplier
        price = model.predict(features)
        adjusted_price = price[0] * price_multiplier
        st.success(f"Estimated House Price: ${adjusted_price:,.2f}")
    except ValueError as ve:
        st.error(f"Invalid input for prediction: {ve}")
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")

    