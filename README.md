## Name : M.Harini
## Reg No: 212222240035
## Date :

# Exp.no.09  A project on Time series analysis on Microsoft Stock Prediction using ARIMA model 

## AIM:
To Create a project on Time series analysis on Microsoft Stock Prediction using ARIMA model in  Python and compare with other models.
## ALGORITHM:
1. Explore the dataset of Microsoft stock prediction
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
## PROGRAM:
```
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
data = pd.read_csv('tsla_2014_2023 (1).csv')
# Ensure the date column is in the correct format; adjust the column name as needed
data['date'] = pd.to_datetime(data['date'])  
data.set_index('date', inplace=True)

# Visualize the time series data to inspect trends
plt.figure(figsize=(10, 5))
plt.plot(data['close'], label='Stock Price')  # Adjust column name as needed, e.g., 'Close'
plt.title('Time Series of Stock Price')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Check stationarity using the ADF test
result = adfuller(data['close'])  # Adjust column name as needed
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# If p-value > 0.05, apply differencing to achieve stationarity
data['close_diff'] = data['close'].diff().dropna()  # Adjust column name as needed
result_diff = adfuller(data['close_diff'].dropna())
print('Differenced ADF Statistic:', result_diff[0])
print('Differenced p-value:', result_diff[1])

# Plot ACF and PACF for the differenced data
plot_acf(data['close_diff'].dropna())
plt.title('ACF of Differenced Stock Price')
plt.show()

plot_pacf(data['close_diff'].dropna())
plt.title('PACF of Differenced Stock Price')
plt.show()

# Plot Differenced Representation
plt.figure(figsize=(10, 5))
plt.plot(data['close_diff'], label='Differenced Stock Price', color='red')
plt.title('Differenced Representation of Stock Price')
plt.xlabel('Date')
plt.ylabel('Differenced Stock Price')
plt.axhline(0, color='black', lw=1, linestyle='--')
plt.legend()
plt.show()

# Use auto_arima to find the optimal (p, d, q) parameters
stepwise_model = auto_arima(data['close'], start_p=1, start_q=1,
                            max_p=3, max_q=3, seasonal=False, trace=True)
p, d, q = stepwise_model.order
print(stepwise_model.summary())

# Fit the ARIMA model using the optimal parameters
model = sm.tsa.ARIMA(data['close'], order=(p, d, q))
fitted_model = model.fit()
print(fitted_model.summary())

# Forecast the next 30 days (or adjust period based on your needs)
forecast = fitted_model.forecast(steps=30)
forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')

# Plot actual vs forecasted values
plt.figure(figsize=(12, 6))
plt.plot(data['close'], label='Actual Stock Price')
plt.plot(forecast_index, forecast, label='Forecast', color='orange')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('ARIMA Forecast of Stock Price')
plt.legend()
plt.show()

# Evaluate the model with MAE and RMSE
predictions = fitted_model.predict(start=0, end=len(data['close']) - 1)
mae = mean_absolute_error(data['close'], predictions)
rmse = np.sqrt(mean_squared_error(data['close'], predictions))
print('Mean Absolute Error (MAE):', mae)
print('Root Mean Squared Error (RMSE):', rmse)
```

## OUTPUT:

### Time series of Stock price:

![Screenshot 2024-11-11 114051](https://github.com/user-attachments/assets/3d87a2c2-e0c9-42db-be1c-e70b73e90a79)


### ACF anf PACF Representation:
![Screenshot 2024-11-11 114058](https://github.com/user-attachments/assets/8eb57751-c7fb-4570-905c-1b7f434dda86)

![Screenshot 2024-11-11 114106](https://github.com/user-attachments/assets/158ab34c-c123-40ce-9a78-9d3d0554d331)

![Screenshot 2024-11-11 114115](https://github.com/user-attachments/assets/d3e59e5e-3fae-4d1c-89f2-481ba47e1025)


### Differencing Representation of Stock price :

![Screenshot 2024-11-11 114128](https://github.com/user-attachments/assets/e20d4705-d0fc-4af2-b076-b5ad40ae7b11)

### Model Summary :
![Screenshot 2024-11-11 114140](https://github.com/user-attachments/assets/e773b7a6-a1e8-4bee-a81f-ea0ee294d4ab)

### ARIMA forecast of stock price :
![Screenshot 2024-11-11 114158](https://github.com/user-attachments/assets/d5b08007-0ddb-4f9b-a2f9-90a13aecf5ce)


## RESULT:
Thus the project on Time Series Analysis on Microsoft Stock prediction based on the ARIMA model using python is executed Successfully.
