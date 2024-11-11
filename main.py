import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load crime rate data from CSV
crime_data = pd.read_csv('crime_data.csv')  # Replace 'crime_data.csv' with your actual file name
years = crime_data['Year'].values.reshape(-1, 1)
crime_rates = crime_data['CrimeRate'].values.reshape(-1, 1)

# Standardize the features (optional)
scaler = StandardScaler()
years_scaled = scaler.fit_transform(years)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(years_scaled, crime_rates, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create and train a Ridge regression model
ridge_model = Ridge(alpha=1.0)  # You can adjust the alpha parameter for different levels of regularization
ridge_model.fit(X_train, y_train)

# Extend the X_test range to cover all years and scale it
X_test_full_range = np.arange(min(years), max(years) + 1).reshape(-1, 1)
X_test_full_range_scaled = scaler.transform(X_test_full_range)

# Make predictions on the full range of years
crime_predictions = ridge_model.predict(X_test_full_range_scaled)

# Evaluate the model on the validation set
mse_valid = mean_squared_error(y_valid, ridge_model.predict(X_valid))
print(f'Mean Squared Error on Validation Set: {mse_valid}')

# Evaluate the model on the test set
mse_test = mean_squared_error(y_test, ridge_model.predict(X_test))
print(f'Mean Squared Error on Test Set: {mse_test}')

# Plot only the line connecting actual crime rates without individual dots
plt.figure(figsize=(12, 6))
plt.plot(X_test_full_range, crime_predictions, label='Predicted Crime Rates (Ridge Regression)', color='red')
plt.plot(years, crime_rates, 'o-', color='blue', markersize=0, label='Connected Actual Crime Rates')
plt.xlabel('Year (Standardized)', fontsize=20)
plt.ylabel('Crime Rate', fontsize=20)
plt.title('Ridge Regression of Crime Rates in America (1990-2022)', fontsize=24)
plt.legend()
plt.show()