import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import numpy as np

h_size = 128
o_size = 128
num_letter = 34
embed_dim = 512


class LAS(nn.Module):
    def __init__(self):
        super(LAS, self).__init__()
        global h_size
        self.listener = Listener()
        self.speller = Speller()

        self.embedding = nn.Embedding(num_letter, embed_dim)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, utter_list, targets):
        b_size = len(utter_list)
        hk, hv, h_lens = self.listener(utter_list)

        y_targets = []  # the target list to be returned(skip position 0)
        emb_targets = []
        for t in targets:
            y_targets.append(t[1:])
            t = self.embedding(t)
            emb_targets.append(t)
        packed_targets = rnn.pad_sequence(emb_targets)  # shape (max(l), b_size, emb-dim)

        sh = init((b_size, o_size))
        sc = init((b_size, o_size))
        c = torch.zeros(b_size, o_size)
        in_mask = get_mask(h_lens)
        predictions = torch.zeros((b_size, len(packed_targets) - 1, num_letter))

        sh = sh.to(self.device)
        sc = sc.to(self.device)
        c = c.to(self.device)
        in_mask = in_mask.to(self.device)
        predictions = predictions.to(self.device)
        
        for idx in range(len(packed_targets) - 1):  # -1 because we do not use } as input
            y_in = packed_targets[idx]  # shape is (b_size, emb_dim)

            # take in target, c, sh, sc for this step and return pred, c, sh, sc for next step
            # pred should be compared with the target at next timestamp
            pred, c, sh, sc = self.speller(hk, hv, y_in, c, sh, sc, in_mask)
            predictions[:, idx, :] = pred

        return predictions, y_targets


# the listener
class Listener(nn.Module):
    def __init__(self):
        super(Listener, self).__init__()
        global h_size
        global o_size
        self.rnn = nn.LSTM(input_size=40, hidden_size=h_size, num_layers=1, bidirectional=True)
        self.conv1 = nn.Conv1d(in_channels=2 * h_size, out_channels=h_size, kernel_size=2, stride=2)
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


# the speller with attention
class Speller(nn.Module):
    def __init__(self):
        super(Speller, self).__init__()
        global o_size
        self.rnnCell = nn.LSTMCell((o_size + embed_dim), o_size)

        self.sm = nn.Softmax(dim=1)

        self.linear1 = nn.Linear(o_size, num_letter)
        self.linear2 = nn.Linear(o_size, num_letter)
        self.activate = nn.ReLU()
        self.linear3 = nn.Linear(num_letter, num_letter)

    # _1 means params from time t-1
    # return values are all for time t
    def forward(self, hk, hv, y_1, c_1, sh_1, sc_1, mask):
        global o_size

        # calculate state for t
        sh, sc = self.rnnCell(torch.cat((y_1, c_1), dim=1), (sh_1, sc_1))

        # calculate attention
        temp = torch.bmm(hk, sc.reshape(sc.size(0), sc.size(1), 1))
        temp = temp.reshape(temp.size(0), temp.size(1))
        temp = self.sm(temp)

        temp = temp * mask
        temp /= torch.sum(temp, dim=1).reshape(-1, 1)

        attention = temp.reshape(temp.size(0), 1, temp.size(1))

        c = torch.bmm(attention, hv)
        c = c.reshape((c.size(0), c.size(2)))

        # calculate prediction
        out = self.linear1(sh)
        out += self.linear2(c)
        out = self.activate(out)
        out = self.linear3(out)
        out = self.sm(out)
        return out, c, sh, sc


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


def get_mask(h_lens):
    global o_size
    b_size = len(h_lens)
    mask = torch.zeros((b_size, h_lens[0]))
    for i in range(b_size):
        mask[i][:h_lens[i]] = 1

    return mask


if __name__ == '__main__':
    pass
