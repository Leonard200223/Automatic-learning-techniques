import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load crime rate data from CSV
crime_data = pd.read_csv('crime_data.csv')  # Replace 'crime_data.csv' with your actual file name
years = crime_data['Year'].values.reshape(-1, 1)
crime_rates = crime_data['CrimeRate'].values

# Create a binary target variable based on some threshold (e.g., median)
threshold = np.median(crime_rates)
crime_labels = (crime_rates > threshold).astype(int)

# Standardize the features (optional)
scaler = StandardScaler()
years_scaled = scaler.fit_transform(years)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(years_scaled, crime_labels, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create polynomial features
degree = 2  # You can adjust the degree of the polynomial
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_valid_poly = poly_features.transform(X_valid)
X_test_poly = poly_features.transform(X_test)

# Create and train a Logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_poly, y_train)

# Extend the X_test range to cover all years and scale it
X_test_full_range = np.arange(min(years), max(years) + 1).reshape(-1, 1)
X_test_full_range_scaled = scaler.transform(X_test_full_range)

# Plot decision boundary
plt.figure(figsize=(12, 6))
plt.scatter(X_test_full_range, crime_labels, color='blue', label='Actual Crime Rates')
plt.xlabel('Year (Standardized)', fontsize=20)
plt.ylabel('Crime Rate (Binary)', fontsize=20)
plt.title(f'Logistic Regression with Polynomial Features (Degree {degree}) of Crime Rates in America (1990-2022)', fontsize=24)

# Plot decision boundary
X_range_poly = poly_features.transform(X_test_full_range_scaled)
decision_boundary = logistic_model.predict(X_range_poly)
plt.plot(X_test_full_range, decision_boundary, label='Decision Boundary', color='red')

plt.legend()
plt.show()

# Evaluate the model on the validation set
accuracy_valid = accuracy_score(y_valid, logistic_model.predict(X_valid_poly))
confusion_matrix_valid = confusion_matrix(y_valid, logistic_model.predict(X_valid_poly))
print(f'Accuracy on Validation Set: {accuracy_valid}')
print(f'Confusion Matrix on Validation Set:\n{confusion_matrix_valid}')

# Evaluate the model on the test set
accuracy_test = accuracy_score(y_test, logistic_model.predict(X_test_poly))
confusion_matrix_test = confusion_matrix(y_test, logistic_model.predict(X_test_poly))
print(f'Accuracy on Test Set: {accuracy_test}')
print(f'Confusion Matrix on Test Set:\n{confusion_matrix_test}')
