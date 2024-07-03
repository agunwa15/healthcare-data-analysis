import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# Load the dataset
file_path = 'files/HSPAH24.20240701121945.csv'
data = pd.read_csv(file_path)

# Display basic information about the dataset
data.info()
data.head()

# DATA AND PREPARATION

# Convert TLIST(M1) to datetime format
data['TLIST(M1)'] = pd.to_datetime(data['TLIST(M1)'].astype(str), format='%Y%m')

# Check for anomalies in the VALUE column
value_stats = data['VALUE'].describe()

# Display cleaned data and statistics for VALUE column
data_cleaned_head = data.head()
print(data_cleaned_head)
print(value_stats)

# EXPLORATORY DATA ANALYSIS (EDA)

# Plotting Readmissions Over Time
plt.figure(figsize=(12, 6))
plt.plot(data['TLIST(M1)'], data['VALUE'], marker='o')
plt.title('Surgical Readmissions Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Readmissions')
plt.grid(True)
plt.show()

# Summary Statistics
print(value_stats)

# Visualizing Distribution of Readmissions
plt.figure(figsize=(12, 6))
sns.histplot(data['VALUE'], bins=15, kde=True)
plt.title('Distribution of Surgical Readmissions')
plt.xlabel('Number of Readmissions')
plt.ylabel('Frequency')
plt.show()

# MODEL DEVELOPMENT

# Stationarity Check
result = adfuller(data['VALUE'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Differencing to make the series stationary if necessary
data['VALUE_diff'] = data['VALUE'].diff().dropna()

# Model Identification (use AIC/BIC for parameter selection)
model = ARIMA(data['VALUE_diff'].dropna(), order=(1, 1, 1))
model_fit = model.fit()

# Model Summary
model_summary = model_fit.summary()
print(model_summary)

# Residuals of the model
residuals = model_fit.resid

# Plot residuals
plt.figure(figsize=(12, 6))
plt.plot(residuals)
plt.title('Residuals of ARIMA Model')
plt.show()

# Plot ACF and PACF of residuals
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
plot_acf(residuals, ax=ax[0])
plot_pacf(residuals, ax=ax[1])
plt.show()

# Forecast future values
forecast_steps = 12  # Forecasting for the next 12 months

# Get the last date in the dataset
last_date = data['TLIST(M1)'].max()


# Generate forecast dates correctly
forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='M')

# Plot the forecasted values
plt.figure(figsize=(12, 6))
plt.plot(data['TLIST(M1)'], data['VALUE'], label='Original')
plt.plot(forecast_dates, model_fit.forecast(steps=forecast_steps), label='Forecast')
plt.title('Surgical Readmissions Forecast')
plt.xlabel('Date')
plt.ylabel('Number of Readmissions')
plt.legend()
plt.show()
