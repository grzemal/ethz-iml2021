import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

import keras.backend as K
import tensorflow as tf

# load the training data
train_dataset = pd.read_csv('train.csv')
# load the test data
test_dataset = pd.read_csv('test.csv')


# input size of sequence
# input_size


def get_model(device):
    return SimpleNetwork().to(device)


class SimpleNetwork(nn.Module):
    def __init__(self, input_size=80):
        super(SimpleNetwork, self).__init__()
        # self.classifier = nn.Sequential(
        #     nn.Linear(input_size, 80),
        #     nn.BatchNorm1d(80),
        #     nn.Linear(80, 80),
        #     nn.BatchNorm1d(80),
        #     nn.Linear(80, 1)
        # )
        self.layer_1 = nn.Linear(input_size, 80)
        self.layer_2 = nn.Linear(80, 80)
        self.layer_out = nn.Linear(80, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.batchnorm1 = nn.BatchNorm1d(80)
        self.batchnorm2 = nn.BatchNorm1d(80)

    def forward(self, input):
        # logits = self.classifier(x)
        x = self.relu(self.layer_1(input))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.sigmoid(self.layer_out(x))
        return x


def experiment(num_epochs, training_data, device, model, optimizer):
    train_results = {}
    test_results = {}
    # Initial test error
    loss, acc, time = test(device, model)
    print(f'Upon initialization. [Test] \t Time {time.avg:.2f} \
            Loss {loss.avg:.2f} \t Accuracy {acc.avg:.2f}')
    test_results[0] = (loss, acc, time)

    for epoch in range(1, num_epochs + 1):
        loss, acc, time = train(training_data, device, model, optimizer)
        print(f'Epoch {epoch}. [Train] \t Time {time.sum:.4f} Loss \
                {loss.avg:.4f} \t Accuracy {acc.avg:.4f}')
        train_results[epoch] = (loss.avg, acc.avg, time.avg)

        if not (epoch % 2):
            loss, acc, time = test(device, model)
            print(f'Epoch {epoch}. [Test] \t Time {time.sum:.4f} Loss \
                {loss.avg:.4f} \t Accuracy {acc.avg:.4f}')
            test_results[epoch] = (loss.avg, acc.avg, time.avg)

    return train_results, test_results


def train(training_set, device, model, optimizer):
    time_ = AverageMeter()
    loss_ = AverageMeter()
    acc_ = AverageMeter()
    model.train()

    for train_x_data, train_labels in train_loader:
        # Accounting
        end = time.time()
        # get the inputs; data is a list of [inputs, labels]
        # train_x_data, train_labels = data  # bs x 3 x 32 x 32
        train_x_data = train_x_data.to(device)
        train_labels = train_labels.to(device)
        bs = train_x_data.size(0)
        # zero the parameter gradients
        optimizer.zero_grad()  # all the tensors have .grad attribute
        # forward propagation
        prediction = model(train_x_data)  # forward propagation
        loss = criterion(prediction, train_labels.unsqueeze(1))  # computing the loss for predictions
        # Backward propagation
        loss.backward()  # backpropagation
        # Optimization step.
        optimizer.step()  # applying an optimization step

        # Accounting
        acc = (torch.round(prediction) == train_labels.unsqueeze(1)).sum() / bs
        loss_.update(loss.mean().item(), bs)
        acc_.update(acc.item(), bs)
        time_.update(time.time() - end)

    return loss_, acc_, time_


def test(device, model):
    time_ = AverageMeter()
    loss_ = AverageMeter()
    acc_ = AverageMeter()
    model.eval()
    #
    # for i in range(len(test_data[1])):
    #     # Accounting
    #     end = time.time()
    #
    #     test_x_data, test_labels = test_data
    #
    #     test_x_data = test_x_data.to(device)
    #     test_labels = test_labels.to(device)
    #
    #     bs = test_x_data.size(0)

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            end = time.time()
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            bs = X_batch.size(0)
            y_test_prediction = model(X_batch)
            loss = criterion(y_test_prediction, y_batch.unsqueeze(1))
            acc = (torch.round(y_test_prediction) == y_batch.unsqueeze(1)).sum() / bs
            # Accounting
            loss_.update(loss.mean().item(), bs)
            acc_.update(acc.mean().item(), bs)
            time_.update(time.time() - end)
    return loss_, acc_, time_


class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def diffable_f1_loss(y_pred, y_true):  # from https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


################### Settings and Execution of Experiment ######################

criterion = nn.BCELoss()
# criterion = diffable_f1_loss();

torch.manual_seed(42)
used_device = torch.device("cpu")
model = get_model(used_device)
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 30
batch_size = 100

################### END Settings and Execution of Experiment ######################


# Format für train_data:  x_data, labels = train_data

# One Hot Encoding of datasets
# Train dataset
x_data = pd.get_dummies(pd.DataFrame(train_dataset['Sequence'].apply(list).to_list(),
                                     columns=['1', '2', '3', '4']), dtype=float)

# for test dataset split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    x_data.to_numpy(), train_dataset['Active'].to_numpy(), test_size=0.33, random_state=42)

# X_train = torch.tensor(X_train, dtype=torch.float32)
# X_test = torch.tensor(X_test, dtype=torch.float32)

# y_train = torch.tensor(y_train, dtype=torch.float32)
# y_test = torch.tensor(y_test, dtype=torch.float32)

# train_data = (X_train, y_train)
# test_data = (X_test, y_test)

# Test dataset
test_dataset = pd.get_dummies(pd.DataFrame(test_dataset['Sequence'].apply(list).to_list(),
                                           columns=['1', '2', '3', '4']))
test_dataset = torch.tensor(test_dataset.to_numpy(), dtype=torch.float32)
from torch.utils.data import Dataset, DataLoader


class trainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


train_data = trainData(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
train_loader = DataLoader(dataset=train_data, batch_size=batch_size)


# test data
class testData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


test_data = trainData(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
test_loader = DataLoader(dataset=test_data, batch_size=1)

train_res, test_res = experiment(num_epochs=num_epochs,
                                 training_data=train_data,
                                 device=used_device,
                                 model=model,
                                 optimizer=optimizer)

from sklearn.metrics import f1_score
y_predicted = torch.round(model(torch.FloatTensor(X_test))).reshape(-1).detach().numpy()
y_true = y_test
print(f1_score(y_true, y_predicted))

prediction = torch.round(model(test_dataset)).reshape(-1).detach().numpy()
# save to csv
np.savetxt("predictions.csv", prediction, fmt="%i")
