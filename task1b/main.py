import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge

fileTrain = "train.csv"

#config
functions = 21

#read data
dataFrame = pd.read_csv(fileTrain)
y = dataFrame.y.to_numpy()
xi = dataFrame.iloc[:, 2:].to_numpy()

# list of features
def feature_description(x):
    return [x[0], x[1], x[2], x[3], x[4],
            x[0]**2, x[1]**2, x[2]**2, x[3]**2, x[4]**2,
            np.exp(x[0]), np.exp(x[1]), np.exp(x[2]), np.exp(x[3]), np.exp(x[4]),
            np.cos(x[0]), np.cos(x[1]), np.cos(x[2]), np.cos(x[3]), np.cos(x[4]),
            1]

feature_matrix = np.array([feature_description(xi[i]) for i in range(len(xi))])

#Linear regression
reg = LinearRegression(fit_intercept=False)
reg.fit(feature_matrix, y)
#print(reg.score(feature_matrix, y))
#print(reg.coef_)

#SDG regression
reg = SGDRegressor(loss='huber', fit_intercept=False, random_state=42)
reg.fit(feature_matrix, y)
#print(reg.score(feature_matrix, y))
#print(reg.coef_)

#Ridge regression with built-in cross-validation.
#reg = RidgeCV(fit_intercept=False)
#reg.fit(feature_matrix, y)
#print(reg.score(feature_matrix, y))
#print(reg.coef_)

#Ridge regression
#reg = Ridge(alpha=0.01, fit_intercept=False)
#reg.fit(feature_matrix, y)
#print(reg.score(feature_matrix, y))
#print(reg.coef_)

#save output
np.savetxt("output.csv",reg.coef_,delimiter=",")
