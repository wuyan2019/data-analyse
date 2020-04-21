import matplotlib.pyplot as plt
import time
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

from model import LSTMSeq2Seq
from config import *

sns.set()
torch.manual_seed(1234)


def generate(data):
    inputs = []
    outputs = []
    for i in range(len(data) - timestamp):
        inputs.append(list(data[i:i + timestamp, 4:9]))
        label = (data[i + timestamp, 5] - data[i + timestamp - 1, 5]) / data[i + timestamp - 1, 5]
        if label > limit:
            outputs.append(0)
        elif -limit < label < limit:
            outputs.append(1)
        else:
            outputs.append(2)
    print('Number of seqs: {}'.format(len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float).to(device), torch.tensor(outputs).to(device))
    return dataset


def calculate_accuracy(real, predict):
    # real = np.array(real) + 1
    # predict = np.array(predict) + 1
    # percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
    # return percentage
    total = len(real)
    true_postive = 0
    for i in range(total):
        if real[i] == predict[i]:
            true_postive += 1
    return true_postive / total



def train(x_train):
    dataloader = DataLoader(generate(x_train.values), batch_size=batch_size,
                            shuffle=True)
    model = LSTMSeq2Seq(input_size=feature_size, hidden_size=hidden_size,
                        num_layers=num_layers, output_size=output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()  # the target label is not one-hotted
    bar = tqdm(range(num_epochs), desc='train loop')
    best_loss = None
    for epoch in bar:  # Loop over the dataset multiple times
        train_loss = 0
        for step, (seq, output_data) in enumerate(dataloader):
            seq = seq.view(-1, timestamp, feature_size)
            output_pre = model(seq)
            loss = criterion(output_pre, output_data)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        if best_loss is None:
            best_loss = train_loss
        if train_loss < best_loss:
            torch.save(model.state_dict(), model_path)
            best_loss = train_loss
        bar.set_postfix(epoch=epoch + 1, loss=train_loss / len(dataloader))
    print('Finished Training')


def test(x_test):
    predict_seq = []
    target = []
    start_time = time.time()
    dataloader = DataLoader(generate(x_test.values), batch_size=1,
                            shuffle=True)
    model = LSTMSeq2Seq(input_size=feature_size, hidden_size=hidden_size,
                        num_layers=num_layers, output_size=output_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        for step, (seq, label) in enumerate(dataloader):
            seq = seq.view(-1, timestamp, feature_size)
            predict_out = model(seq)
            predict_seq.append(predict_out.argmax())
            target.append(label)
    acc = calculate_accuracy(target, predict_seq)
    print('Accuracy is {}, test time is {}.'.format(acc, time.time() - start_time))
    # fig = plt.figure()
    # # plt.plot(x_train, label='train')
    # plt.plot(x_test.iloc[:, 5], label='true trend')
    # gain = []
    # for i in range(1, len(predict_seq)):
    #     gain.append((1 + predict_seq[i]) * x_test.iloc[i + timestamp - 1, 5])
    # plt.plot(list(x_test.index)[timestamp + 1:], gain, label='predict trend')
    # fig.legend(loc='upper left')
    # plt.show()


if __name__ == '__main__':
    df = pd.read_csv(data_path)
    df = df.loc[:2440]
    # Select the 'close' for the forecast
    # minmax = MinMaxScaler().fit(df.iloc[:, 5:7].astype('float32'))
    # df_log = minmax.transform(df.iloc[:, 5:7].astype('float32'))
    # df_log = pd.DataFrame(df_log)
    print('Shape', df.shape)
    split_size = 2200
    x_train = df[:split_size]
    x_test = df[split_size:]
    if model_train:
        train(x_train)
    else:
        test(x_test)
