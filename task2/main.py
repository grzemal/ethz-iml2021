import pandas as pd
import numpy as np
import time

# load the training data
train_features = pd.read_csv('train_features.csv').\
    sort_values(by=['pid', 'Time'])
train_labels = pd.read_csv('train_labels.csv').\
    sort_values(by=['pid'])

# load the test data
test_features = pd.read_csv('test_features.csv')

# balance imbalanced data

def make_features(raw_data):
    n_meas = raw_data.groupby('pid', sort=False).count()

    # handle missing data
    raw_data = raw_data.fillna(raw_data.groupby(['pid'], sort=False).ffill())
    raw_data = raw_data.fillna(raw_data.groupby(['pid'], sort=False).bfill())
    raw_data = raw_data.fillna(raw_data.median())

    last_meas = raw_data.groupby('pid', sort=False).last()
    median = raw_data.groupby('pid', sort=False).median()
    
    interquartile_range = raw_data.groupby('pid', sort=False).\
        quantile([.25, .75]).groupby('pid', sort=False).\
        diff().groupby('pid', sort=False).first(1)

    frames = [n_meas.iloc[:, 1:], last_meas.iloc[:, 1:],
              median.iloc[:, 1:], interquartile_range.iloc[:, 1:]]
    features = pd.concat(frames, axis=1, join="inner")
    return features

start = time.time()
X = make_features(train_features)
X_test = make_features(test_features)
end = time.time()
print(end - start)

y = train_labels

##########################################################################
#############################  Sub-task 1  ###############################
##########################################################################
labels_sub_task1 = ["LABEL_BaseExcess", "LABEL_Fibrinogen", "LABEL_AST",
                    "LABEL_Alkalinephos", "LABEL_Bilirubin_total",
                    "LABEL_Lactate", "LABEL_TroponinI", "LABEL_SaO2",
                    "LABEL_Bilirubin_direct", "LABEL_EtCO2"]

prediction_sub_task1 = np.zeros((X_test.to_numpy().shape[0],
                                 len(labels_sub_task1)))

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

start = time.time()
clf = HistGradientBoostingClassifier(random_state=42)
for i in range(len(labels_sub_task1)):
    clf.fit(X.to_numpy(), y[[labels_sub_task1[i]]].to_numpy().flatten())
    prediction_sub_task1[:, i] = clf.predict_proba(X_test.to_numpy())[:, 1]
end = time.time()
print(end - start)

##########################################################################
#############################  Sub-task 2  ###############################
##########################################################################
labels_sub_task2 = ["LABEL_Sepsis"]
prediction_sub_task2 = np.zeros((X_test.to_numpy().shape[0],
                                 len(labels_sub_task2)))

start = time.time()
clf = HistGradientBoostingClassifier(random_state=42)
for i in range(len(labels_sub_task2)):
    clf.fit(X.to_numpy(), y[[labels_sub_task2[i]]].to_numpy().flatten())
    prediction_sub_task2[:, i] = clf.predict_proba(X_test.to_numpy())[:, 1]
end = time.time()
print(end - start)

##########################################################################
#############################  Sub-task 3  ###############################
##########################################################################
from sklearn.ensemble import HistGradientBoostingRegressor

labels_sub_task3 = ["LABEL_RRate", "LABEL_ABPm", "LABEL_SpO2",
                    "LABEL_Heartrate"]

prediction_sub_task3 = np.zeros((X_test.to_numpy().shape[0],
                                 len(labels_sub_task3)))
start = time.time()
est = HistGradientBoostingRegressor(random_state=42)
for i in range(len(labels_sub_task3)):
    est.fit(X.to_numpy(), y[[labels_sub_task3[i]]].to_numpy().flatten())
    prediction_sub_task3[:, i] = est.predict(X_test.to_numpy())
end = time.time()
print(end - start)

##########################################################################
#############################  Submission  ###############################
##########################################################################
d = {'pid': X_test.index.values,
     labels_sub_task1[0]: prediction_sub_task1[:, 0],
     labels_sub_task1[1]: prediction_sub_task1[:, 1],
     labels_sub_task1[2]: prediction_sub_task1[:, 2],
     labels_sub_task1[3]: prediction_sub_task1[:, 3],
     labels_sub_task1[4]: prediction_sub_task1[:, 4],
     labels_sub_task1[5]: prediction_sub_task1[:, 5],
     labels_sub_task1[6]: prediction_sub_task1[:, 6],
     labels_sub_task1[7]: prediction_sub_task1[:, 7],
     labels_sub_task1[8]: prediction_sub_task1[:, 8],
     labels_sub_task1[9]: prediction_sub_task1[:, 9],

     labels_sub_task2[0]: prediction_sub_task2[:, 0],

     labels_sub_task3[0]: prediction_sub_task3[:, 0],
     labels_sub_task3[1]: prediction_sub_task3[:, 1],
     labels_sub_task3[2]: prediction_sub_task3[:, 2],
     labels_sub_task3[3]: prediction_sub_task3[:, 3]
     }
df = pd.DataFrame(data=d)

df.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')
df.to_csv('prediction.csv', index=False, float_format='%.3f')