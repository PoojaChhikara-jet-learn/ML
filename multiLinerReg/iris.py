import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add target variable (species) for reference
df['species'] = iris.target

# Display the first few rows of the dataset
print(df.head())

# Prepare the data: using sepal length and sepal width as features
X = df[["sepal length (cm)", "sepal width (cm)"]]
Y = df["petal length (cm)"]  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

# Multivariable Regression
lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

# Predict on the test set
y_test_predict = lin_model.predict(X_test)

# Calculate RMSE for multivariable regression
rmse_lin_model = np.sqrt(mean_squared_error(Y_test, y_test_predict))
print("The RMSE in case of multivariable regression is ", rmse_lin_model)

# Polynomial Regression
poly_feature = PolynomialFeatures(degree=2)

# Transform the training data
X_train_poly = poly_feature.fit_transform(X_train)
poly_model = LinearRegression()
poly_model.fit(X_train_poly, Y_train)

# Transform the test data and predict
X_test_poly = poly_feature.transform(X_test)
y_test_predict_poly = poly_model.predict(X_test_poly)

# Calculate RMSE for polynomial regression
rmse_poly_model = np.sqrt(mean_squared_error(Y_test, y_test_predict_poly))
print("The RMSE in case of polynomial regression is ", rmse_poly_model)

# Optional: Visualize the predictions
plt.figure(figsize=(12, 6))

# Plot for Linear Regression
plt.subplot(1, 2, 1)
sns.scatterplot(x=Y_test, y=y_test_predict)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
plt.title('Multivariable Regression Predictions')
plt.xlabel('Actual Petal Length')
plt.ylabel('Predicted Petal Length')

# Plot for Polynomial Regression
plt.subplot(1, 2, 2)
sns.scatterplot(x=Y_test, y=y_test_predict_poly)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
plt.title('Polynomial Regression Predictions')
plt.xlabel('Actual Petal Length')
plt.ylabel('Predicted Petal Length')

plt.tight_layout()
plt.show()
