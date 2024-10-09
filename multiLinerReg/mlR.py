import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston
# from sklearn.datasets import fetch_california_housing
# housing = fetch_california_housing()

# # boston_data = load_boston()
# from sklearn.datasets import fetch_openml
# housing = fetch_openml(name="house_prices", as_frame=True)

boston = pd.DataFrame(boston_data.data, columns = boston_data.feature_names)
print(boston.head())


boston["MEDV"] = boston_data.target

#sns.set(rc = {"figure.figsize" : (11.7, 8.27)})
#sns.displot(boston["MEDV"], bins = 30)
#plt.show()

# Prepare the data from train-test split

X = boston[["LSTAT", "RM"]]
Y = boston["MEDV"]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 5)

# Mutlivariable Regression
from sklearn.linear_model import LinearRegression
lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

from sklearn.metrics import mean_squared_error
y_test_predict = lin_model.predict(X_test)

rmse_lin_model = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
print("The RMSE in case of multivariable regression is ", rmse_lin_model)

from sklearn.preprocessing import PolynomialFeatures
poly_feature = PolynomialFeatures(degree = 2)

X_train_poly = poly_feature.fit_transform(X_train)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, Y_train)

X_test_poly = poly_feature.fit_transform(X_test)
y_test_predict_poly = poly_model.predict(X_test_poly)

rmse_poly_model = (np.sqrt(mean_squared_error(Y_test, y_test_predict_poly)))
print("The RMSE in case of multivariable regression is ", rmse_poly_model)