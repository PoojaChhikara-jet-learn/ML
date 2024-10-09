import numpy as np

# Sample dataset or we can use csv files
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 4, 9, 16, 25])

# Calculate the slope (m) and intercept (b)
n = len(x)
x_mean = np.mean(x)
y_mean = np.mean(y)

numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)

m = numerator / denominator
b = y_mean - m * x_mean

# Print the equation of the line
print(f"Line of Best Fit: y = {m:.2f}x + {b:.2f}")

# Predict y values based on the model
y_pred = m * x + b

# Calculate RMSE
rmse = np.sqrt(np.mean((y - y_pred) ** 2))
print(f"Root Mean Squared Error: {rmse:.2f}")
