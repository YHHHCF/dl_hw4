import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

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

        sh = []
        sc = []
        sh0 = init((b_size, o_size)).to(self.device)
        sc0 = init((b_size, o_size)).to(self.device)
        sh1 = init((b_size, o_size)).to(self.device)
        sc1 = init((b_size, o_size)).to(self.device)
        sh2 = init((b_size, o_size)).to(self.device)
        sc2 = init((b_size, o_size)).to(self.device)

        sh.append(sh0)
        sh.append(sh1)
        sh.append(sh2)
        sc.append(sc0)
        sc.append(sc1)
        sc.append(sc2)

        c = torch.zeros(b_size, o_size).to(self.device)

        in_mask = get_mask(h_lens)
        in_mask = in_mask.to(self.device)
        predictions = torch.zeros((b_size, len(packed_targets) - 1, num_letter))
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
        self.conv3 = nn.Conv1d(in_channels=h_size, out_channels=h_size, kernel_size=2, stride=2)
        self.rnn3 = nn.LSTM(input_size=h_size, hidden_size=h_size, num_layers=1, bidirectional=False)

        self.kLinear = nn.Linear(h_size, o_size)
        self.vLinear = nn.Linear(h_size, o_size)

        self.bn = nn.BatchNorm1d(h_size)

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
        padded_output = self.bn(padded_output)
        padded_output = trans(padded_output, False)
        padded_output, _ = self.rnn1(padded_output)

        padded_output = trans(padded_output, True)
        padded_output = self.conv2(padded_output)
        padded_output = self.bn(padded_output)
        padded_output = trans(padded_output, False)
        padded_output, _ = self.rnn2(padded_output)

        padded_output = trans(padded_output, True)
        padded_output = self.conv3(padded_output)
        padded_output = self.bn(padded_output)
        padded_output = trans(padded_output, False)
        padded_output, _ = self.rnn3(padded_output)

        padded_output = torch.transpose(padded_output, 0, 1)
        lens = (lens / 8)

        k = self.kLinear(padded_output)
        v = self.vLinear(padded_output)

        return k, v, lens


# the speller with attention
class Speller(nn.Module):
    def __init__(self):
        super(Speller, self).__init__()
        global o_size
        self.rnnCell0 = nn.LSTMCell((o_size + embed_dim), o_size)
        self.rnnCell1 = nn.LSTMCell(o_size, o_size)
        self.rnnCell2 = nn.LSTMCell(o_size, o_size)

        self.sm = nn.Softmax(dim=1)

        self.linear1 = nn.Linear(o_size, h_size)
        self.linear2 = nn.Linear(o_size, h_size)
        self.activate = nn.ReLU()
        self.linear3 = nn.Linear(h_size, num_letter)

        self.bn = nn.BatchNorm1d(h_size)

    # _1 means params from time t-1
    # return values are all for time t
    def forward(self, hk, hv, y_1, c_1, sh_1, sc_1, mask):
        global o_size

        sh = []
        sc = []

        # calculate state for t
        sh0, sc0 = self.rnnCell0(torch.cat((y_1, c_1), dim=1), (sh_1[0], sc_1[0]))
        # sh0 = self.bn(sh0)
        # sc0 = self.bn(sc0)
        sh1, sc1 = self.rnnCell1(sh0, (sh_1[1], sc_1[1]))
        # sh1 = self.bn(sh1)
        # sc1 = self.bn(sc1)
        sh2, sc2 = self.rnnCell2(sh1, (sh_1[2], sc_1[2]))
        # sh2 = self.bn(sh2)
        # sc2 = self.bn(sc2)

        sh.append(sh0)
        sh.append(sh1)
        sh.append(sh2)
        sc.append(sc0)
        sc.append(sc1)
        sc.append(sc2)

        # calculate attention
        temp = torch.bmm(hk, sc2.reshape(sc2.size(0), sc2.size(1), 1))
        temp = temp.reshape(temp.size(0), temp.size(1))
        temp = self.sm(temp)

        temp = temp * mask
        temp /= torch.sum(temp, dim=1).reshape(-1, 1)

        attention = temp.reshape(temp.size(0), 1, temp.size(1))

        c = torch.bmm(attention, hv)
        c = c.reshape((c.size(0), c.size(2)))

        # calculate prediction
        out = self.linear1(sh2)
        out += self.linear2(c)
        out = self.bn(out)
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
