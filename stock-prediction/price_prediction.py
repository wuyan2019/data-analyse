import matplotlib.pyplot as plt
import time
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import LSTMSeq2Seq
from config import *

sns.set()
torch.manual_seed(1234)


def generate(data):
    inputs = []
    outputs = []
    for i in range(len(data) - timestamp):
        inputs.append(data[i:i + timestamp])
        outputs.append(data[i + timestamp])
    print('Number of seqs: {}'.format(len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return dataset


def calculate_accuracy(real, predict):
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
    return percentage


def train(x_train):
    dataloader = DataLoader(generate(x_train.values), batch_size=batch_size,
                            shuffle=True, pin_memory=True)
    model = LSTMSeq2Seq(input_size=timestamp, hidden_size=hidden_size,
                        num_layers=num_layers, output_size=output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()  # the target label is not one-hotted
    bar = tqdm(range(num_epochs), desc='train loop')
    best_loss = None
    for epoch in bar:  # Loop over the dataset multiple times
        train_loss = 0
        for step, (seq, output_data) in enumerate(dataloader):
            seq = seq.view(-1, 1, timestamp)
            output_pre = model(seq)
            loss = criterion(output_pre.view(-1, 1), output_data)
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
    start_time = time.time()
    dataloader = DataLoader(generate(x_test.values), batch_size=1,
                            shuffle=True, pin_memory=True)
    model = LSTMSeq2Seq(input_size=timestamp, hidden_size=hidden_size,
                        num_layers=num_layers, output_size=output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        for step, (seq, label) in enumerate(dataloader):
            seq = seq.view(-1, 1, timestamp)
            predict_out = model(seq)
            predict_seq.append(predict_out)
    acc = calculate_accuracy(x_test.values[timestamp:], predict_seq)
    print('Accuracy is {}, test time is {}.'.format(acc, time.time() - start_time))
    fig = plt.figure()
    plt.plot(x_train, label='train')
    plt.plot(x_test, label='true trend')
    plt.plot(list(x_test.index)[5:], predict_seq, label='predict trend')
    fig.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv(data_path)
    # Select the 'close' for the forecast
    minmax = MinMaxScaler().fit(df.iloc[:, 4:5].astype('float32'))
    df_log = minmax.transform(df.iloc[:, 4:5].astype('float32'))
    df_log = pd.DataFrame(df_log)
    print('Shape', df_log.shape)
    x_train = df_log[:split_size]
    x_test = df_log[split_size:]
    if model_train:
        train(x_train)
    else:
        test(x_test)
