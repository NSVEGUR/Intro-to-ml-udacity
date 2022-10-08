from sklearn.linear_model import LinearRegression
import numpy as np


features_train = np.array([[10, 20], [10, 25], [12, 30], [8, 10], [5, 35]])
labels_train = np.array([1, 1, 1, -1, -1])

features_test = np.array([[5, 1], [10, 7]])
labels_test = np.array([-1, 1])

reg = LinearRegression()
reg.fit(features_train, labels_train)
print(f'Coefficients: {reg.coef_}')
print(f'Intercept: {reg.intercept_}')
print("\n##### Stats on Test Data #####\n")
print(f'Score: {reg.score(features_test, labels_test)}')
print("\n##### Stats on Train Data #####\n")
print(f'Score: {reg.score(features_train, labels_train)}')


