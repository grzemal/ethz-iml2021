import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor

fileTrain = "train.csv"

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

#SDG regression
reg = SGDRegressor(loss='squared_epsilon_insensitive', penalty='l2', fit_intercept=False, random_state=0)
reg.fit(feature_matrix, y)

#save output
np.savetxt("output_sgd.csv", reg.coef_, delimiter=",")
