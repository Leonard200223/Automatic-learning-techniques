import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load crime rate data from CSV
crime_data = pd.read_csv('crime_data.csv')  # Replace 'crime_data.csv' with your actual file name
years = crime_data['Year'].values.reshape(-1, 1)
crime_rates = crime_data['CrimeRate'].values.reshape(-1, 1)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(years, crime_rates, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Reshape y_train and y_test to avoid DataConversionWarning
y_train = y_train.ravel()
y_test = y_test.ravel()

# Create and train a Random Forest regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust the number of estimators
rf_model.fit(X_train, y_train)

# Extend the X_test range to cover all years
X_test_full_range = np.arange(min(years), max(years) + 1).reshape(-1, 1)

# Make predictions on the full range of years
crime_predictions = rf_model.predict(X_test_full_range)

# Evaluate the model on the validation set
mse_valid = mean_squared_error(y_valid, rf_model.predict(X_valid))
print(f'Mean Squared Error on Validation Set: {mse_valid}')

# Evaluate the model on the test set
mse_test = mean_squared_error(y_test, rf_model.predict(X_test))
print(f'Mean Squared Error on Test Set: {mse_test}')

# Plot only the line connecting actual crime rates without individual dots
plt.figure(figsize=(12, 6))
plt.plot(X_test_full_range, crime_predictions, label='Predicted Crime Rates (Random Forest)', color='green')

# Connect the actual crime rates with a line and make dots disappear
plt.plot(years, crime_rates, 'o-', color='blue', markersize=0, label='Connected Actual Crime Rates')

plt.xlabel('Year', fontsize=20)
plt.ylabel('Crime Rate', fontsize=20)
plt.title('Random Forest Regression of Crime Rates in America (1990-2022)', fontsize=24)
plt.legend()
plt.show()
