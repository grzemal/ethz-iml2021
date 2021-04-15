import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline

# load the training data
train_features = pd.read_csv('train_features.csv'). \
    sort_values(by=['pid', 'Time'])
train_labels = pd.read_csv('train_labels.csv'). \
    sort_values(by=['pid'])

# load the test data
test_features = pd.read_csv('test_features.csv')

# gridCV
parameters_clf = {
    'scoring': ['roc_auc'],
    'random_state': [42],
    'max_iter': [1000],
    'max_depth': [10, 25, 50],
    'max_bins': [20, 100, 255],
    'max_leaf_nodes': [10, 50, 100]
 }
parameters_est = {
    'scoring': ['r2'],
    'random_state': [42],
    'max_iter': [1000],
    'max_depth': [10, 25, 50],
    'max_bins': [20, 100, 255],
    'max_leaf_nodes': [10, 50, 100]
 }

# balance imbalanced data

def make_features(raw_data):
    n_meas = raw_data.groupby('pid', sort=False).count()

    # handle missing data
    raw_data = raw_data.fillna(raw_data.groupby(['pid'], sort=False).ffill())
    raw_data = raw_data.fillna(raw_data.groupby(['pid'], sort=False).bfill())
    raw_data = raw_data.fillna(raw_data.median())

    last_meas = raw_data.groupby('pid', sort=False).last()
    median = raw_data.groupby('pid', sort=False).median()

    interquartile_range = raw_data.groupby('pid', sort=False). \
        quantile([.25, .75]).groupby('pid', sort=False). \
        diff().groupby('pid', sort=False).first(1)

    frames = [n_meas.iloc[:, 1:], last_meas.iloc[:, 1:],
              median.iloc[:, 1:], interquartile_range.iloc[:, 1:]]
    features = pd.concat(frames, axis=1, join="inner")
    return features


start = time.time()
rnd_seed = 42
X_test = make_features(test_features)
X_train = make_features(train_features)
y_train = train_labels

# X_test_ = make_features(test_features)
end = time.time()
print(end - start)


##########################################################################
#############################  Sub-task 1  ###############################
##########################################################################
labels_sub_task1 = ["LABEL_BaseExcess", "LABEL_Fibrinogen", "LABEL_AST",
                    "LABEL_Alkalinephos", "LABEL_Bilirubin_total",
                    "LABEL_Lactate", "LABEL_TroponinI", "LABEL_SaO2",
                    "LABEL_Bilirubin_direct", "LABEL_EtCO2"]

X_sub_task12 = StandardScaler().fit_transform(X_train.to_numpy())
X_test_sub_task12 = StandardScaler().fit_transform(X_test.to_numpy())

prediction_sub_task1 = np.zeros((X_test.to_numpy().shape[0],
                                 len(labels_sub_task1)))

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

#clf = HistGradientBoostingClassifier(random_state=rnd_seed)
#pipe = make_pipeline(HistGradientBoostingClassifier())
clf = GridSearchCV(HistGradientBoostingClassifier(), parameters_clf, n_jobs=5,
                   cv=5, scoring='roc_auc', verbose=2, refit=True)

for i in range(len(labels_sub_task1)):
    start = time.time()
    clf.fit(X_train.to_numpy(), y_train[[labels_sub_task1[i]]].to_numpy().flatten())
    prediction_sub_task1[:, i] = clf.predict_proba(X_test.to_numpy())[:, 1]
    #print("Score for ", labels_sub_task1[i], " :",
    #      clf.score(X_test.to_numpy(), y_test[[labels_sub_task1[i]]].to_numpy().flatten()))
    print("Score: ", clf.best_score_)
    print("Best parameters :", clf.best_params_)
    end = time.time()
    print("Time to compute :", end - start)

##########################################################################
#############################  Sub-task 2  ###############################
##########################################################################
labels_sub_task2 = ["LABEL_Sepsis"]
prediction_sub_task2 = np.zeros((X_test.to_numpy().shape[0],
                                 len(labels_sub_task2)))

#clf = HistGradientBoostingClassifier(random_state=rnd_seed)

for i in range(len(labels_sub_task2)):
    start = time.time()
    clf.fit(X_train.to_numpy(), y_train[[labels_sub_task2[i]]].to_numpy().flatten())
    prediction_sub_task2[:, i] = clf.predict_proba(X_test.to_numpy())[:, 1]
    #print("Score for ", labels_sub_task2[i], " :",
    #      clf.score(X_test.to_numpy(), y_test[[labels_sub_task2[i]]].to_numpy().flatten()))
    print("Score: ", clf.best_score_)
    print("Best parameters :", clf.best_params_)
    end = time.time()
    print("Time to compute :", end - start)

##########################################################################
#############################  Sub-task 3  ###############################
##########################################################################
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

labels_sub_task3 = ["LABEL_RRate", "LABEL_ABPm", "LABEL_SpO2",
                    "LABEL_Heartrate"]

prediction_sub_task3 = np.zeros((X_test.to_numpy().shape[0],
                                 len(labels_sub_task3)))

#est = HistGradientBoostingRegressor(random_state=rnd_seed)
#pipe = make_pipeline(HistGradientBoostingRegressor())
est = GridSearchCV(HistGradientBoostingRegressor(), parameters_est, n_jobs=5,
                   cv=5, scoring='r2', verbose=2, refit=True)

for i in range(len(labels_sub_task3)):
    start = time.time()
    est.fit(X_train.to_numpy(), y_train[[labels_sub_task3[i]]].to_numpy().flatten())
    prediction_sub_task3[:, i] = est.predict(X_test.to_numpy())
    #print("Score for ", labels_sub_task3[i], " :",
    #      est.score(X_test.to_numpy(), y_test[[labels_sub_task3[i]]].to_numpy().flatten()))
    end = time.time()
    print("Score: ", est.best_score_)
    print("Best parameters :", est.best_params_)
    print("Time to compute :", end - start)

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