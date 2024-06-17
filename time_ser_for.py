import pandas as pd
import numpy as np

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
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the time series data
plt.figure(figsize=(14, 7))
plt.plot(data['Global_active_power'])
plt.title('Global Active Power over Time')
plt.xlabel('Time')
plt.ylabel('Global Active Power (kilowatts)')
#plt.show()

# Plot the correlations between different features
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
#plt.show()

# Summary statistics
#print(data.describe())

# Create time-based features
data['Hour'] = data.index.hour
data['Day'] = data.index.day
data['Month'] = data.index.month
data['Day_of_week'] = data.index.dayofweek
data['Weekend'] = (data.index.dayofweek >= 5).astype(int)

# Optional: Normalize or scale the features if required
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[['Global_active_power', 'Global_reactive_power', 'Voltage', 
                                             'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 
                                             'Sub_metering_3']])
from statsmodels.tsa.arima.model import ARIMA

# Example: ARIMA Model
model = ARIMA(data['Global_active_power'], order=(5, 1, 0))
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

# Plot the residuals
residuals = pd.DataFrame(model_fit.resid)
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals')
plt.show()
