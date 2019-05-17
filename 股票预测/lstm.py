import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta


sns.set()
df = pd.read_csv('GOOG-year.csv')
date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
df.head()

minmax = MinMaxScaler().fit(df.iloc[:, 1:].astype('float32'))
df_log = minmax.transform(df.iloc[:, 1:].astype('float32'))
df_log = pd.DataFrame(df_log)
df_log.head()


class Model:
    def __init__(
        self,
        learning_rate,
        num_layers,
        size,
        size_layer,
        output_size,
        forget_bias = 0.1,
    ):
        def lstm_cell(size_layer):
            return torch.nn.LSTMCell(size_layer, state_is_tuple=False)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(size_layer) for _ in range(num_layers)]
        )
        self.X = tf.placeholder(tf.float32, (None, None, size))
        self.Y = tf.placeholder(tf.float32, (None, output_size))
        drop = tf.contrib.rnn.DropoutWrapper(
            rnn_cells, output_keep_prob = forget_bias
        )
        self.hidden_layer = tf.placeholder(
            tf.float32, (None, num_layers * 2 * size_layer)
        )
        self.outputs, self.last_state = tf.nn.dynamic_rnn(
            drop, self.X, initial_state = self.hidden_layer, dtype = tf.float32
        )
        self.logits = tf.layers.dense(self.outputs[-1], output_size)
        # self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.loss_fn, lr=learning_rate)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        #     self.cost
        # )


class ModelClass(torch.nn.Module):
    def __init__(self, num_layers, size, size_layer, output_size):
        super(ModelClass, self).__init__()
        self.lstm = torch.nn.LSTM(size, size_layer, batch_first=True)
        self.drop = torch.nn.Dropout(0.5)
        self.linear1 = torch.nn.Linear(num_layers * 2 * size_layer, output_size)

    def forward(self, x):
        x = self.lstm(x)
        x = self.drop(x)
        x = self.linear1(x)
        return x


class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=torch.nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out=torch.nn.Linear(in_features=64,out_features=10)

    def forward(self,x):
        # 一下关于shape的注释只针对单项
        # output: [batch_size, time_step, hidden_size]
        # h_n: [num_layers,batch_size, hidden_size] # 虽然LSTM的batch_first为True,但是h_n/c_n的第一维还是num_layers
        # c_n: 同h_n
        output,(h_n,c_n)=self.rnn(x)
        print(output.size())
        # output_in_last_timestep=output[:,-1,:] # 也是可以的
        output_in_last_timestep=h_n[-1,:,:]
        # print(output_in_last_timestep.equal(output[:,-1,:])) #ture
        x=self.out(output_in_last_timestep)
        return x


# Initialize model
net = RNN()

# Initialize optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
loss_F = torch.nn.MSELoss()

num_layers = 1
size_layer = 128
timestamp = 5
epoch = 500
dropout_rate = 0.7
future_day = 50

for step, input_data in enumerate(df['Close']):
    x, y = input_data, input_data

    pred=net(x)
    break;
    loss=loss_F(pred,y) # 计算loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step%50==49: # 每50步，计算精度
        with torch.no_grad():
            test_pred=net(test_x)
            prob=torch.nn.functional.softmax(test_pred,dim=1)
            pred_cls=torch.argmax(prob,dim=1)
            acc=(pred_cls==test_y).sum().numpy()/pred_cls.size()[0]
            print(f"{epoch}-{step}: accuracy:{acc}")