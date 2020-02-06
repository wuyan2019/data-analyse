import torch
import torch.nn as nn


class LstmModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LstmModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, dropout=0.05, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.lstm(x)   # None 表示 hidden state 会用全0的 state

        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        output = self.out(r_out[:, -1, :])
        return output


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers, dropout=0.05, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        r_out, (h_n, h_c) = self.gru(x)
        output = self.out(r_out[:, -1, :])
        return output


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers, dropout=0.05, batch_first=True)

    def forward(self, x):
        output, hidden = self.gru(x)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Decoder, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers, dropout=0.05, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        output, hidden = self.gru(x, hidden)
        output = self.out(output)
        return output, hidden


class LSTMSeq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMSeq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(input_size, hidden_size, num_layers, output_size)

    def forward(self, src):

        encoder_out, hidden = self.encoder(src)
        decoder_out, hidden = self.decoder(src, hidden)
        return decoder_out
