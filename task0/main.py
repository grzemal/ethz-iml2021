import pandas as pd
from sklearn.metrics import mean_squared_error

fileTrain = "train.csv"
fileTest = "test.csv"

# Step 1: Train
dataFrame = pd.read_csv(fileTrain)
Idx = dataFrame.Id
y = dataFrame.y
xi = dataFrame.iloc[:, 2:]
y_pred = xi.mean(axis = 1)

# Step 2: Evaluation
RMSE = mean_squared_error(y, y_pred)**0.5
print(RMSE)

# Step 3: Test
dataFrame = pd.read_csv(fileTest)
Idx = dataFrame.Id
xi = dataFrame.iloc[:, 1:]
y_pred = xi.mean(axis = 1)

# Step 4: Save output
output = pd.DataFrame({'Id': Idx, 'y': y_pred})
output.to_csv('output.csv', index = False)

