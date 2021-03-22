import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import numpy as np

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

#linear regression
clf = LinearRegression(fit_intercept=False,n_jobs=-1)
reg = clf.fit(feature_matrix, y)

#save output
np.savetxt("output_new.csv",reg.coef_,delimiter=",")
