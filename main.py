import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

salary_df = pd.read_csv("salary_data.csv")

salary_df

salary_df = pd.get_dummies(salary_df, ["Gender", "Job Title", "Country", "Race"], drop_first = True)

X = salary_df.drop(columns=["Salary"])
y = salary_df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

regr = linear_model.LinearRegression()

regr.fit(X_train, y_train)

LinearRegression()

y_pred = regr.predict(X_test)

print('Coefficients: \n', regr.coef_)
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred, color='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color = 'blue', linewidth = 3)

plt.xlabel('Actual Values')
plt.ylabel('Prediction Values')

plt.show()


salary_df = pd.read_csv("salary_data.csv")
salary_df = pd.get_dummies(salary_df, ["Gender", "Job Title", "Country", "Race"], drop_first = True) 

X = salary_df.drop(columns = ["Salary"]) 
y = salary_df["Salary"] 

poly = PolynomialFeatures(degree = 2) 
X_poly = poly.fit_transform(X) 

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size = 0.2) 

ridge = Ridge(alpha = 1.0) 
ridge.fit(X_train, y_train) 
y_pred = ridge.predict(X_test) 

print('Coefficients: \n', ridge.coef_) 
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred)) 
print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred)) 

plt.scatter(y_test, y_pred, color='black') 
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color ='blue', linewidth=3) 
plt.xlabel('Actual Values') 
plt.ylabel('Prediction Values') 

plt.show()


