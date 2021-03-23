import pandas as pd
import numpy as np

from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

fileTrain = "train.csv"

#config
#lambdas = np.linspace(0.01, 1000.0, num=10000)
rand_int = 42
test_size_per = 0.15

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

#x_train, x_test, y_train, y_test = train_test_split(feature_matrix, y, test_size=test_size_per, random_state=rand_int)

#SDG regression
reg = SGDRegressor(loss='squared_loss', penalty='l1', fit_intercept=False, random_state=rand_int)
reg.fit(feature_matrix, y)
#print(reg.coef_)
#print(reg.n_iter_)

#RMSE
#prediction = reg.predict(x_test)
#RMSE = mean_squared_error(y_test, prediction) ** 0.5
#print(RMSE)


#Ridge regression with built-in cross-validation.
#reg = RidgeCV(alphas = lambdas, fit_intercept=False)
#reg.fit(feature_matrix, y)
#print(reg.score(feature_matrix, y))
#print(reg.alpha_)

#save output
np.savetxt("output.csv",reg.coef_,delimiter=",")
