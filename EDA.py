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
plt.show()

# Plot the correlations between different features
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Summary statistics
print(data.describe())
