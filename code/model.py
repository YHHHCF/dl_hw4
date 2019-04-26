import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import numpy as np


class LAS(nn.Module):
    def __init__(self):
        super(LAS, self).__init__()
        self.listener = Listener()
        self.atten = Attention()
        self.speller = Speller()

        self.embedding = nn.Embedding(34, 64)

    def forward(self, utter_list, targets):
        print("db:", utter_list[0].shape)
        k, v, lens = self.listener(utter_list)

        for target in targets:
            mask_len = len(target)
            s1 = init((3, 3))
            s2 = init((3, 3))
            target = self.embedding(target)  # shape = (mask_len, embed_dim)

            for idx in range(len(target)):
                # get attention and mask it
                c = self.atten(k, v, s1, s2)
                # go through speller and get output
                out = self.speller(target[idx], c, s1, s2)

        return out


# the listener
class Listener(nn.Module):
    def __init__(self):
        super(Listener, self).__init__()
        h_size = 20
        o_size = 3
        self.rnn = nn.LSTM(input_size=40, hidden_size=h_size, num_layers=1, bidirectional=False)
        self.conv1 = nn.Conv1d(in_channels=h_size, out_channels=h_size, kernel_size=2, stride=2)
        self.rnn1 = nn.LSTM(input_size=h_size, hidden_size=h_size, num_layers=1, bidirectional=False)
        self.conv2 = nn.Conv1d(in_channels=h_size, out_channels=h_size, kernel_size=2, stride=2)
        self.rnn2 = nn.LSTM(input_size=h_size, hidden_size=h_size, num_layers=1, bidirectional=False)

        self.kLinear = nn.Linear(h_size, o_size)
        self.vLinear = nn.Linear(h_size, o_size)

    def forward(self, utter_list):  # list
        # concat input on dim=0
        # packed_input[0] has shape (sum(l), 40), dtype=float
        # packed_input[1] has shape (max(l)), dtype=int
        packed_input = rnn.pack_sequence(utter_list)

        # packed_output[0] has shape (sum(l), hidden_size or *2), dtype=float
        # packed_output[1] has shape (max(l)), dtype=int
        packed_output, hidden = self.rnn(packed_input, None)

        # shape (max(l), batch_size, hidden_size or *2), dtype=float
        # padded_output[:,i,:] (shape is (max(l), hidden_size or *2)) corresponds to the i'th input
        # lens are lengths for each tensor before padding(after padding will all be max(l))
        padded_output, lens = rnn.pad_packed_sequence(packed_output)

        # go through pBLSTM
        padded_output = trans(padded_output, True)
        padded_output = self.conv1(padded_output)
        padded_output = trans(padded_output, False)
        padded_output, _ = self.rnn1(padded_output)
        padded_output = trans(padded_output, True)
        padded_output = self.conv2(padded_output)
        padded_output = trans(padded_output, False)
        padded_output, _ = self.rnn2(padded_output)

        padded_output = torch.transpose(padded_output, 0, 1)
        lens = (lens / 4)

        k = self.kLinear(padded_output)
        v = self.vLinear(padded_output)

        return k, v, lens


def trans(data, flag):
    if flag:
        data = torch.transpose(data, 0, 1)
        data = torch.transpose(data, 1, 2)
    else:
        data = torch.transpose(data, 1, 2)
        data = torch.transpose(data, 0, 1)
    return data


def init(shape):
    tensor = torch.zeros(shape)
    nn.init.xavier_uniform_(tensor)
    return tensor


# attention algorithm
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.sm = nn.Softmax(dim=1)  # debug

    # k, v corresponds to h and s1, s2 corresponds to s
    def forward(self, k, v, s1, s2):
        print("k", k.shape)
        c = torch.mean(v, dim=1)  # use average now instead of attention
        # s = s1 * s2  # to be debug
        # ek = torch.bmm(k, s)
        # print("ek", ek.shape)
        # ek = self.sm(ek)

        return c


# the speller
class Speller(nn.Module):
    def __init__(self):
        super(Speller, self).__init__()
        self.linear1 = nn.Linear(1, 1)
        self.linear2 = nn.Linear(1, 1)
        self.linear3 = nn.Linear(1, 1)
        self.linear4 = nn.Linear(1, 1)

        self.rnnCell = nn.LSTMCell(3, 32)

    def forward(self, y_1, c_1, s1_1, s2_1):
        print(y_1.shape)
        print(c_1.shape)
        print(s1_1.shape)
        print(s2_1.shape)
        s1, s2 = self.rnnCell(torch.cat(y_1, c_1), (s1_1, s2_1))
        out = self.linear1(s1)
        out += self.linear2(s2)
        out += self.linear3(c_1)
        out = self.linear4(out)
        return out


if __name__ == '__main__':
    pass
