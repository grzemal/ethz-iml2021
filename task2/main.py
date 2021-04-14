import pandas as pd
import numpy as np
from scipy.stats import iqr

# load the training data
train_features = pd.read_csv('train_features.csv').sort_values(by=['pid', 'Time'])
train_labels = pd.read_csv('train_labels.csv').sort_values(by=['pid'])

# load the test data
test_features = pd.read_csv('test_features.csv')

x_ = train_features.iloc[:, 2:]


def create_features(raw_data):
    number_of_features_per_channel = 4
    tot_mean = raw_data.mean(axis=0)
    new_features = np.zeros((int(raw_data.shape[0] / 12), raw_data.shape[1] * number_of_features_per_channel))
    for i in range(int(raw_data.shape[0] / 12)):
        patient_features = raw_data.iloc[12 * i:12 * (i + 1), :]
        patient_features_ffill = patient_features.ffill(axis=0)

        last = len(patient_features_ffill.values) - 1
        #patient_features_ffill.iloc[last, :] = patient_features_ffill.iloc[last, :].fillna(tot_mean)
        patient_features_ffill = patient_features_ffill.fillna(tot_mean)

        n_meas = 12 - patient_features.isna().sum().to_numpy()
        last_meas = patient_features_ffill.iloc[last, :].to_numpy()
        median = patient_features_ffill.median(axis=0).to_numpy()
        interquartile_range = iqr(patient_features_ffill.to_numpy(), axis=0)

        new_single_features = np.concatenate((n_meas, last_meas, median, interquartile_range))
        new_features[i, :] = new_single_features

    return new_features
