import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
data = pd.read_csv('household_power_consumption.txt', sep=';', 
                   na_values=['?'],
                   low_memory=False)

# Parse dates manually
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S', dayfirst=True)

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

# Set the Datetime column as the index and ensure it's sorted
data.set_index('Datetime', inplace=True)
data.sort_index(inplace=True)

# Set the frequency of the datetime index
data.index = pd.to_datetime(data.index).to_period('min')

# Optional: Normalize or scale the features if required
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[['Global_active_power', 'Global_reactive_power', 'Voltage', 
                                             'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 
                                             'Sub_metering_3']])

# Reduce the dataset size for fitting the ARIMA model (optional)
data_sampled = data.sample(frac=0.1, random_state=1)  # Use 10% of the data

# Fit the ARIMA model
model = ARIMA(data_sampled['Global_active_power'], order=(5, 1, 0))
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

# Plot the residuals and save to a file
residuals = pd.DataFrame(model_fit.resid)
plt.figure(figsize=(10, 6))
plt.plot(residuals.index.to_timestamp(), residuals)  # Convert PeriodIndex to Timestamp for plotting
plt.title('Residuals')
plt.savefig('residuals.png')
plt.close()

# Predictions
train_size = int(len(data_sampled) * 0.8)
train, test = data_sampled['Global_active_power'][:train_size], data_sampled['Global_active_power'][train_size:]
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()
predictions = model_fit.forecast(steps=len(test))

# Evaluation
rmse = np.sqrt(mean_squared_error(test, predictions))
mae = mean_absolute_error(test, predictions)
mape = np.mean(np.abs((test - predictions) / test)) * 100

print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'MAPE: {mape}%')

# Plot the predictions vs actual values and save to a file
plt.figure(figsize=(14, 7))
plt.plot(test.index.to_timestamp(), test, label='Actual Data')
plt.plot(test.index.to_timestamp(), predictions, label='Predicted Data', color='red')
plt.title('Actual vs Predicted Global Active Power')
plt.xlabel('Time')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.savefig('predictions_vs_actual.png')
plt.close()

# Forecast future consumption
future_steps = 30  # Example: next 30 minutes
forecast = model_fit.forecast(steps=future_steps)
forecast_dates = pd.date_range(start=data.index[-1].to_timestamp(), periods=future_steps+1, closed='right')

# Plot the forecast and save to a file
plt.figure(figsize=(14, 7))
plt.plot(data['Global_active_power'].index.to_timestamp(), data['Global_active_power'], label='Historical Data')
plt.plot(forecast_dates, forecast, label='Forecasted Data', color='red')
plt.title('Forecast of Global Active Power')
plt.xlabel('Time')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.savefig('forecast.png')
plt.close()
