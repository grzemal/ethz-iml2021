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
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 80),
            nn.BatchNorm1d(80),
            nn.Linear(80, 80),
            nn.BatchNorm1d(80),
            nn.Linear(80, 1)
        )

    def forward(self, x):
        logits = self.classifier(x)
        return logits


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
        print(f'Epoch {epoch}. [Train] \t Time {time.sum:.2f} Loss \
                {loss.avg:.2f} \t Accuracy {acc.avg:.2f}')
        train_results[epoch] = (loss.avg, acc.avg, time.avg)

        if not (epoch % 2):
            loss, acc, time = test(device, model)
            print(f'Epoch {epoch}. [Test] \t Time {time.sum:.2f} Loss \
                {loss.avg:.2f} \t Accuracy {acc.avg:.2f}')
            test_results[epoch] = (loss.avg, acc.avg, time.avg)

    return train_results, test_results


def train(training_set, device, model, optimizer):
    time_ = AverageMeter()
    loss_ = AverageMeter()
    acc_ = AverageMeter()
    model.train()

    for i, data, in enumerate(training_set, 1):
        # Accounting
        end = time.time()

        # get the inputs; data is a list of [inputs, labels]
        train_x_data, train_labels = data  # bs x 3 x 32 x 32
        train_x_data = train_x_data.to(device)
        train_labels = train_labels.to(device)
        bs = train_x_data.size(0)
        # zero the parameter gradients
        optimizer.zero_grad()  # all the tensors have .grad attribute
        # forward propagation
        logits = model(train_x_data)  # forward propagation
        loss = criterion(logits, train_labels)  # computing the loss for predictions
        # Backward propagation
        loss.backward()  # backpropagation
        # Optimization step.
        optimizer.step()  # applying an optimization step

        # Accounting
        acc = (torch.argmax(logits, dim=-1) == train_labels).sum() / bs
        loss_.update(loss.mean().item(), bs)
        acc_.update(acc.item(), bs)
        time_.update(time.time() - end)

    return loss_, acc_, time_


def test(device, model):
    time_ = AverageMeter()
    loss_ = AverageMeter()
    acc_ = AverageMeter()
    model.eval()

    for i in range(len(test_data[1])):
        # Accounting
        end = time.time()

        test_x_data, test_labels = test_data
        print(test_labels.shape)
        test_x_data = test_labels.to(device)
        test_labels = test_labels.to(device)

        bs = test_x_data.size(0)

        with torch.no_grad():
            logits = model(test_x_data)
            loss = criterion(logits, test_labels)
            acc = (torch.argmax(logits, dim=-1) == test_labels).sum() / bs

            # Accounting
            loss_.update(loss.mean().item(), bs)
            acc_.update(acc.mean().item(), bs)
            time_.update(time.time() - end)
            print("finisched test")

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

criterion = nn.CrossEntropyLoss()
# criterion = diffable_f1_loss();

torch.manual_seed(42)
used_device = "cpu"
model = get_model(used_device)
optimizer = optim.SGD(model.parameters(), lr=0.001)

num_epochs = 10
batch_size = 100


################### END Settings and Execution of Experiment ######################


# Format für train_data:  x_data, labels = train_data

# One Hot Encoding of datasets
# Train dataset
x_data = pd.get_dummies(pd.DataFrame(train_dataset['Sequence'].apply(list).to_list(),
                                     columns=['1', '2', '3', '4']))

# for test dataset split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    x_data.to_numpy(), train_dataset['Active'].to_numpy(), test_size=0.33, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_data = (X_train, y_train)
test_data = (X_test, y_test)
# Test dataset
#test_data = pd.get_dummies(pd.DataFrame(test['Sequence'].apply(list).to_list(),
#                                        columns=['1', '2', '3', '4']))
#test_data = torch.tensor(test_data.to_numpy(), dtype=torch.float32)


train_res, test_res = experiment(num_epochs=num_epochs,
                                 training_data=train_data,
                                 device=used_device,
                                 model=model,
                                 optimizer=optimizer)
