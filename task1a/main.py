import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import numpy as np

fileTrain = "train.csv"

#config
folds = 10
regularization_params = np.array([0.1, 1.0, 10.0, 100.0, 200.0])


# Step 1: Train
dataFrame = pd.read_csv(fileTrain)
y = dataFrame.y.to_numpy()
xi = dataFrame.iloc[:, 1:].to_numpy()

#data acquisition and partitioning
kf = KFold(n_splits=folds)
RMSE = [0 for i in range(len(regularization_params))]

#cross validation
for idx_train, idx_test in kf.split(xi):
    #train data
    x_train = xi[idx_train]
    y_train = y[idx_train]

    #test data
    x_test = xi[idx_test]
    y_test = y[idx_test]

    for i in range(len(regularization_params)):
        # ridge regression
        clf = Ridge(regularization_params[i])
        clf.fit(x_train, y_train)
        prediction = clf.predict(x_test)
        prediction_error = mean_squared_error(y_test, prediction) ** 0.5
        RMSE[i] += prediction_error / (1.0 * folds)

#output
np.savetxt("output.csv", RMSE)