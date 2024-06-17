# streamlit_app.py

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Function to load and preprocess the dataset
@st.cache_data
def load_data():
    # Load the dataset
    data = pd.read_csv('household_power_consumption.txt', sep=';', 
                       parse_dates={'Datetime': ['Date', 'Time']},
                       infer_datetime_format=True, 
                       na_values=['?'],
                       low_memory=False)
    
    # Handle missing values
    data.replace('?', np.nan, inplace=True)
    data.dropna(inplace=True)
    
    # Convert columns to appropriate data types
    data['Global_active_power'] = data['Global_active_power'].astype(float)
    data['Global_reactive_power'] = data['Global_reactive_power'].astype(float)
    data['Voltage'] = data['Voltage'].astype(float)
    data['Global_intensity'] = data['Global_intensity'].astype(float)
    data['Sub_metering_1'] = data['Sub_metering_1'].astype(float)
    data['Sub_metering_2'] = data['Sub_metering_2'].astype(float)
    data['Sub_metering_3'] = data['Sub_metering_3'].astype(float)
    
    # Set the Datetime column as the index
    data.set_index('Datetime', inplace=True)
    
    return data

# Load the data
data = load_data()

# Streamlit app
st.title('Household Electricity Consumption Forecast')

# Display raw data
st.subheader('Raw Data')
if st.checkbox('Show raw data'):
    st.write(data.head())

# Exploratory Data Analysis
st.subheader('Exploratory Data Analysis')

# Plot the time series data
st.write("### Global Active Power over Time")
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(data['Global_active_power'])
ax.set_title('Global Active Power over Time')
ax.set_xlabel('Time')
ax.set_ylabel('Global Active Power (kilowatts)')
st.pyplot(fig)

# Plot the correlations between different features
st.write("### Correlation Matrix")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Matrix')
st.pyplot(fig)

# Feature Engineering
st.subheader('Feature Engineering')

# Create time-based features
data['Hour'] = data.index.hour
data['Day'] = data.index.day
data['Month'] = data.index.month
data['Day_of_week'] = data.index.dayofweek
data['Weekend'] = (data.index.dayofweek >= 5).astype(int)

# Optional: Display feature-engineered data
if st.checkbox('Show feature-engineered data'):
    st.write(data.head())

# Model fitting and evaluation
st.subheader('Time Series Forecasting')

# Select ARIMA model parameters
st.write("### Model Training and Evaluation")
p = 5
d = 1
q = 0

# Train-test split
train_size = int(len(data) * 0.8)
train, test = data['Global_active_power'][:train_size], data['Global_active_power'][train_size:]

# Model training
model = ARIMA(train, order=(p, d, q))
model_fit = model.fit()

# Model summary
st.write("#### Model Summary")
st.text(model_fit.summary())

# Predictions and evaluation
predictions = model_fit.forecast(steps=len(test))
rmse = np.sqrt(mean_squared_error(test, predictions))
st.write(f'#### RMSE: {rmse}')

# Plot the residuals
st.write("#### Residuals")
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(residuals)
ax.set_title('Residuals')
st.pyplot(fig)

# Plot actual vs predicted
st.write("#### Actual vs Predicted")
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(test.index, test, label='Actual')
ax.plot(test.index, predictions, label='Predicted', color='red')
ax.set_title('Actual vs Predicted')
ax.set_xlabel('Time')
ax.set_ylabel('Global Active Power (kilowatts)')
ax.legend()
st.pyplot(fig)

# Forecast future consumption
st.subheader('Forecast Future Consumption')
st.write("### Forecast Future Consumption")

with st.form("forecast_form"):
    p = st.number_input('Select p:', min_value=0, max_value=10, value=5)
    d = st.number_input('Select d:', min_value=0, max_value=2, value=1)
    q = st.number_input('Select q:', min_value=0, max_value=10, value=0)
    future_steps = st.slider('Forecast steps into the future:', 1, 365, 30)
    submitted = st.form_submit_button("Forecast")

    if submitted:
        # Retrain the model with the user-selected parameters
        model = ARIMA(data['Global_active_power'], order=(p, d, q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=future_steps)
        forecast_dates = pd.date_range(start=data.index[-1], periods=future_steps + 1, closed='right')

        # Plot the forecast
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(data['Global_active_power'], label='Historical Data')
        ax.plot(forecast_dates, forecast, label='Forecasted Data', color='red')
        ax.set_title('Forecast of Global Active Power')
        ax.set_xlabel('Time')
        ax.set_ylabel('Global Active Power (kilowatts)')
        ax.legend()
        st.pyplot(fig)
