import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge
import numpy as np

fileTrain = "train.csv"

#config
random_seed = 42
folds = 10
regularization_params = np.array([0.1, 1.0, 10.0, 100.0, 200.0])


# Step 1: Train
dataFrame = pd.read_csv(fileTrain)
y = dataFrame.y.to_numpy()
xi = dataFrame.iloc[:, 1:].to_numpy()

#data acquisition and partitioning
rkf = RepeatedKFold(n_splits=folds, n_repeats=10000, random_state=random_seed)
RMSE = [0 for i in range(len(regularization_params))]
current_RMSE = [0 for i in range(len(regularization_params))]

counter = 0


#cross validation
for idx_train, idx_test in rkf.split(xi):
    #train data
    x_train = xi[idx_train]
    y_train = y[idx_train]

    #test data
    x_test = xi[idx_test]
    y_test = y[idx_test]


    for i in range(len(regularization_params)):
        # ridge regression
        clf = Ridge(regularization_params[i], tol=1e-12)
        clf.fit(x_train, y_train)
        prediction = clf.predict(x_test)
        prediction_error = mean_squared_error(y_test, prediction) ** 0.5
        current_RMSE[i] += prediction_error / (1.0 * folds)

    counter += 1
    if counter % folds == 0:
        counter = 0
        #print(current_RMSE)
        print(RMSE)
        for i in range(len(regularization_params)):
            if current_RMSE[i] < RMSE[i] or RMSE[i] == 0:
                RMSE[i] = current_RMSE[i]
            current_RMSE[i] = 0


#output
np.savetxt("output_gihug.csv", RMSE)