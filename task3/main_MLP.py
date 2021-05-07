import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GridSearchCV

# load the training data
train = pd.read_csv('train.csv')
# load the test data
test = pd.read_csv('test.csv')

start = time.time()

# Handle imbalanced data
from sklearn.utils import resample

train_majority = train[train.Active == 0]
train_minority = train[train.Active == 1]

train_minority_upsampled = resample(train_minority,
                                    replace=True,
                                    n_samples=len(train_majority),
                                    random_state=42)
train_upsampled = pd.concat([train_majority, train_minority_upsampled])

# One Hot Encoding
x_train = pd.get_dummies(pd.DataFrame(train_upsampled['Sequence'].apply(list).to_list(),
                                      columns=['', '', '', '']))

y_train = train_upsampled["Active"]

x_test = pd.get_dummies(pd.DataFrame(test['Sequence'].apply(list).to_list(),
                                     columns=['', '', '', '']))

end = time.time()
print("Time to compute transformed features :", end - start)

parameters_clf = {
    'hidden_layer_sizes': [(1000,)],
    'max_iter': [500],
    'activation': ['logistic'],
    'random_state': [42],
    'batch_size': [2000],
    'learning_rate_init': [0.01],
    'learning_rate': ['constant'],
    'verbose': [True]
}

from sklearn.neural_network import MLPClassifier

clf = GridSearchCV(estimator=MLPClassifier(), param_grid=parameters_clf,
                   n_jobs=-1, cv=3, scoring='f1', verbose=2, refit=True)

start = time.time()
clf.fit(x_train.to_numpy(), y_train.to_numpy())
print("Score for train dataset: ", clf.best_score_)
print("Best parameters: ", clf.best_params_)
prediction = clf.predict(x_test.to_numpy())
end = time.time()
print("Time to compute: ", end - start)

# save to csv
np.savetxt("predictions.csv", prediction, fmt="%i")
