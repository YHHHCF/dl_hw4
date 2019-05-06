import torch.nn as nn
import torch.nn.utils.rnn as rnn
from config import *
import numpy as np


# the listener
class Listener(nn.Module):
    def __init__(self):
        super(Listener, self).__init__()
        self.rnn = nn.LSTM(input_size=40, hidden_size=h_size, num_layers=1, bidirectional=True)
        self.conv1 = nn.Conv1d(in_channels=2 * h_size, out_channels=h_size, kernel_size=2, stride=2)
        self.rnn1 = nn.LSTM(input_size=h_size, hidden_size=h_size, num_layers=1, bidirectional=True)
        self.conv2 = nn.Conv1d(in_channels=2 * h_size, out_channels=h_size, kernel_size=2, stride=2)
        self.rnn2 = nn.LSTM(input_size=h_size, hidden_size=h_size, num_layers=1, bidirectional=True)
        self.conv3 = nn.Conv1d(in_channels=2 * h_size, out_channels=h_size, kernel_size=2, stride=2)
        self.rnn3 = nn.LSTM(input_size=h_size, hidden_size=h_size, num_layers=1, bidirectional=True)

        self.kLinear = nn.Linear(2 * h_size, o_size)
        self.vLinear = nn.Linear(2 * h_size, o_size)

        self.bn = nn.BatchNorm1d(2 * h_size)

    def forward(self, utter_list):  # list
        # concat input on dim=0
        # packed_input[0] shape (L, 40), dtype=float
        # packed_input[1] shape (L,), dtype=int
        packed_input = rnn.pack_sequence(utter_list)

        # packed_output[0] has shape (sum(l), 2H), dtype=float
        # packed_output[1] has shape (max(l)), dtype=int
        packed_output, _ = self.rnn(packed_input, None)

        # shape (L, B, 2H), dtype=float
        # lens are lengths for each tensor before padding
        padded_output, lens = rnn.pad_packed_sequence(packed_output)

        # go through pBLSTM
        padded_output = trans(padded_output, True)  # shape (B, 2H, L)
        padded_output = self.conv1(padded_output)  # shape (B, 2H, L/2)
        padded_output = trans(padded_output, False)  # shape (L/2, B, 2H)
        padded_output, _ = self.rnn1(padded_output)  # shape (L/2, B, 2H)

        padded_output = trans(padded_output, True)
        padded_output = self.conv2(padded_output)
        padded_output = trans(padded_output, False)
        padded_output, _ = self.rnn2(padded_output)  # shape (L/4, B, 2H)

        padded_output = trans(padded_output, True)
        padded_output = self.conv3(padded_output)
        padded_output = trans(padded_output, False)
        padded_output, _ = self.rnn3(padded_output)   # shape (L/8, B, 2H)

        padded_output = torch.transpose(padded_output, 0, 1)   # shape (B, L/8, 2H)
        lens = (lens / 8)

        k = self.kLinear(padded_output)   # shape (B, L/8, O)
        v = self.vLinear(padded_output)   # shape (B, L/8, O)

        return k, v, lens


# the speller with attention
class Speller(nn.Module):
    def __init__(self):
        super(Speller, self).__init__()

        self.rnnCell0 = nn.LSTMCell((o_size + embed_dim), o_size)
        self.rnnCell1 = nn.LSTMCell(o_size, o_size)

        self.sm = nn.Softmax(dim=1)

        self.linear1 = nn.Linear(o_size, embed_dim)
        self.linear2 = nn.Linear(o_size, embed_dim)
        self.activate = nn.ReLU()
        self.linear3 = nn.Linear(embed_dim, num_letter)

        self.bn = nn.BatchNorm1d(embed_dim)

    # _1 means params from time t-1
    # return values are all for time t
    def forward(self, hk, hv, y_1, c_1, sh_1, sc_1, mask):

        sh = []
        sc = []

        # calculate state for t
        sh0, sc0 = self.rnnCell0(torch.cat((y_1, c_1), dim=1), (sh_1[0], sc_1[0]))
        sh1, sc1 = self.rnnCell1(sh0, (sh_1[1], sc_1[1]))

        sh.append(sh0)
        sh.append(sh1)
        sc.append(sc0)
        sc.append(sc1)

        # calculate attention
        temp = torch.bmm(hk, sh1.reshape(sh1.size(0), sh1.size(1), 1))  # (B, L, O) * (B, O, 1) = (B, L, 1)
        temp = temp.reshape(temp.size(0), temp.size(1))  # (B, L)
        temp = self.sm(temp)  # (B, L)

        temp = temp * mask  # (B, L)
        temp /= torch.sum(temp, dim=1).reshape(-1, 1)  # (B, L)

        atten_vec = temp[0]  # for debug, (L,)

        attention = temp.reshape(temp.size(0), 1, temp.size(1))  # (B, 1, L)

        # get the context
        c = torch.bmm(attention, hv)  # (B, 1, L) * (B, L, O) = (B, 1, O)
        c = c.reshape((c.size(0), c.size(2)))  # (B, O)

        # calculate prediction
        out = self.linear1(sh1)  # (B, E)
        out += self.linear2(c)  # (B, E)
        out = self.bn(out)
        out = self.activate(out)
        out = self.linear3(out)  # (B, V)

        return out, c, sh, sc, atten_vec


def trans(data, flag):
    if flag:
        data = torch.transpose(data, 0, 1)
        data = torch.transpose(data, 1, 2)
    else:
        data = torch.transpose(data, 1, 2)
        data = torch.transpose(data, 0, 1)
    return data


def zero_init(shape):
    tensor = torch.zeros(shape).to(DEVICE)
    return tensor


# init weights using xavier
def init_weights(m):
    if type(m) == nn.Conv1d or type(m) == nn.Linear:
        nn.init.uniform_(m.weight.data, -0.1, 0.1)

    if type(m) == nn.LSTM or type(m) == nn.LSTMCell:
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.uniform_(param, -0.1, 0.1)


def get_mask(h_lens):
    mask = torch.zeros((b_size, h_lens[0]))
    for i in range(b_size):
        mask[i][:h_lens[i]] = 1
    mask = mask.to(DEVICE)
    return mask


def get_emb(array):
    emb_array = torch.zeros((len(array), num_letter)).to(DEVICE)
    for i in range(len(array)):
        emb_array[i][array[i]] = 1

    return emb_array


# append the input char to the list
def append_char(char_list, char):
    if not char_list is None:
        char_list = np.append(char_list, char)
    else:
        char_list = [char]
    return char_list
