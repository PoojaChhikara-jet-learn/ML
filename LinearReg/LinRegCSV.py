import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data from the CSV file
data = pd.read_csv("ml/ice_cream_sales_vs_temp (1).csv")

# Independent variable (temperature)
X = data["Temperature"].values.reshape(-1, 1)

# Dependent variable (ice-cream sales)
y = data["Ice Cream Sales"].values

# Create the Linear Regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Make predictions based on the model
y_pred = model.predict(X)

# Print the slope (coefficient) and intercept
print(f"Slope (m): {model.coef_[0]}")
print(f"Intercept (b): {model.intercept_}")

# Calculate the RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plotting the data points and the regression line
plt.scatter(X, y, color="blue", label="Data Points")
plt.plot(X, y_pred, color="red", label="Regression Line")
plt.title("Ice-cream Sales vs Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("Ice-cream Sales ($)")
plt.legend()
plt.grid(True)
plt.show()
