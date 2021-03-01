import pandas as pd
from sklearn.metrics import mean_squared_error

fileTrain = "train.csv"
fileTest = "test.csv"

# Train step
dataFrame = pd.read_csv(fileTrain)
y = dataFrame.y
xi = dataFrame.iloc[:, 2:12]
y_pred = xi.mean(axis = 1)

# Evaluation metric
RMSE = mean_squared_error(y, y_pred)**0.5
print(RMSE)

# Test step
dataFrame = pd.read_csv(fileTest)
Idx = dataFrame.Id
xi = dataFrame.iloc[:, 1:11]
y_pred = xi.mean(axis = 1)

# save output
output = pd.DataFrame({'Id': Idx, 'y': y_pred})
output.to_csv('output.csv', index=False)

