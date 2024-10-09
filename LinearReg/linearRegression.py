import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset 
x = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))  # Reshaped for sklearn
y = np.array([1, 4, 9, 16, 25])

# Create the model
model = LinearRegression()

# Fit the model
model.fit(x, y)

# Make predictions
y_pred = model.predict(x)

# Plot the results
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, y_pred, color='red', label='Line of Best Fit')
plt.title('Linear Regression with sklearn')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Print the slope and intercept
print(f"Slope: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

# Calculate RMSE
rmse = np.sqrt(np.mean((y - y_pred) ** 2))
print(f"Root Mean Squared Error: {rmse:.2f}")
