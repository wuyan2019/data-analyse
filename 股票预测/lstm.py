import math
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.nn import Parameter as P
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta


sns.set()
df = pd.read_csv('./GOOG-year.csv')
date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
df.head()

minmax = MinMaxScaler().fit(df.iloc[:, 1:].astype('float32'))
df_log = minmax.transform(df.iloc[:, 1:].astype('float32'))
df_log = pd.DataFrame(df_log)
df_log.head()


# class LSTM(nn.Module):
#     """
#     An implementation of Hochreiter & Schmidhuber:
#     'Long-Short Term Memory'
#     http://www.bioinf.jku.at/publications/older/2604.pdf
#     Special args:
#
#     dropout_method: one of
#             * pytorch: default dropout implementation
#             * gal: uses GalLSTM's dropout
#             * moon: uses MoonLSTM's dropout
#             * semeniuta: uses SemeniutaLSTM's dropout
#     """
#
#     def __init__(self, input_size, hidden_size, bias=True, dropout=0.0, dropout_method='pytorch'):
#         super(LSTM, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.bias = bias
#         self.dropout = dropout
#         self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
#         self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
#         self.reset_parameters()
#         assert (dropout_method.lower() in ['pytorch', 'gal', 'moon', 'semeniuta'])
#         self.dropout_method = dropout_method
#
#     def sample_mask(self):
#         keep = 1.0 - self.dropout
#         self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep)))
#
#     def reset_parameters(self):
#         std = 1.0 / math.sqrt(self.hidden_size)
#         for w in self.parameters():
#             w.data.uniform_(-std, std)
#
#     def forward(self, x, hidden):
#         do_dropout = self.training and self.dropout > 0.0
#         h, c = hidden
#         h = h.view(h.size(1), -1)
#         c = c.view(c.size(1), -1)
#         x = x.view(x.size(1), -1)
#
#         # Linear mappings
#         preact = self.i2h(x) + self.h2h(h)
#
#         # activations
#         gates = preact[:, :3 * self.hidden_size].sigmoid()
#         g_t = preact[:, 3 * self.hidden_size:].tanh()
#         i_t = gates[:, :self.hidden_size]
#         f_t = gates[:, self.hidden_size:2 * self.hidden_size]
#         o_t = gates[:, -self.hidden_size:]
#
#         # cell computations
#         if do_dropout and self.dropout_method == 'semeniuta':
#             g_t = F.dropout(g_t, p=self.dropout, training=self.training)
#
#         c_t = th.mul(c, f_t) + th.mul(i_t, g_t)
#
#         if do_dropout and self.dropout_method == 'moon':
#             c_t.data.set_(th.mul(c_t, self.mask).data)
#             c_t.data *= 1.0 / (1.0 - self.dropout)
#
#         h_t = th.mul(o_t, c_t.tanh())
#
#         # Reshape for compatibility
#         if do_dropout:
#             if self.dropout_method == 'pytorch':
#                 F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
#             if self.dropout_method == 'gal':
#                 h_t.data.set_(th.mul(h_t, self.mask).data)
#                 h_t.data *= 1.0 / (1.0 - self.dropout)
#
#         h_t = h_t.view(1, h_t.size(0), -1)
#         c_t = c_t.view(1, c_t.size(0), -1)
#         return h_t, (h_t, c_t)


class ModelClass(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(ModelClass, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.drop = torch.nn.Dropout(0.5)
        self.out = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None 表示 hidden state 会用全0的 state

        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        out = self.out(r_out[:, -1, :])
        return out


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(     # LSTM 效果要比 nn.RNN() 好多了
            input_size=200,      # 图片每行的数据像素点
            hidden_size=256,     # rnn hidden unit
            num_layers=1,       # 有几层 RNN layers
            batch_first=True,   # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(256, 52)    # 输出层

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)

        outs = []  # save all predictions
        for time_step in range(r_out.size(1)):  # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)   # optimize all parameters
loss_func = nn.MSELoss()   # the target label is not one-hotted

EPOCH = 2
x = df_log[0][:200]
y = df_log[0][200:]
h_state = None
prediction, h_state = rnn(x, h_state)
h_state = Variable(h_state.data)

loss = loss_func(prediction, y)  # cross entropy loss
optimizer.zero_grad()  # clear gradients for this training step
loss.backward()  # backpropagation, compute gradients
optimizer.step()  # apply gradients
# # Initialize model
# net = RNN()
#
# # Initialize optimizer
# optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# loss_F = torch.nn.MSELoss()
#
# num_layers = 1
# size_layer = 128
# timestamp = 5
# epoch = 500
# dropout_rate = 0.7
# future_day = 50
#
# for step, input_data in enumerate(df['Close']):
#     x, y = input_data, input_data
#
#     pred = net(x)
#     break;
#     loss=loss_F(pred,y) # 计算loss
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if step%50==49: # 每50步，计算精度
#         with torch.no_grad():
#             test_pred=net(test_x)
#             prob=torch.nn.functional.softmax(test_pred,dim=1)
#             pred_cls=torch.argmax(prob,dim=1)
#             acc=(pred_cls==test_y).sum().numpy()/pred_cls.size()[0]
#             print(f"{epoch}-{step}: accuracy:{acc}")
