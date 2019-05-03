import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import numpy as np
from label_proc import *
from queue import PriorityQueue as PQ

b_size = 256

h_size = 128
o_size = 128

num_letter = 34
embed_dim = 256
tf_rate = 0.1

beam_width = 2

beam_alpha = 1.4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LAS(nn.Module):
    def __init__(self):
        super(LAS, self).__init__()
        global h_size
        global b_size
        global tf_rate
        global beam_width
        global DEVICE

        self.listener = Listener()
        self.speller = Speller()

        self.embedding = nn.Embedding(num_letter, embed_dim)

    # mode = train/val/test
    # when training, use tf; when val/test, use beam search
    def forward(self, utter_list, targets, mode):
        global tf_rate
        global b_size
        global DEVICE

        hk, hv, h_lens = self.listener(utter_list)

        if mode == 'train' or mode == 'val':
            y_targets = []  # the target list to be returned(skip position 0)

        emb_targets = []
        for t in targets:
            if mode == 'train' or mode == 'val':
                y_targets.append(t[1:])
            t = self.embedding(t)
            emb_targets.append(t)
        packed_targets = rnn.pad_sequence(emb_targets)  # shape (max(l), b_size, emb-dim)

        # a mask of shape(b_size, max(l)) to mask input attention
        in_mask = get_mask(h_lens, mode=mode)
        in_mask = in_mask.to(DEVICE)

        predictions = torch.zeros((b_size, len(packed_targets) - 1, num_letter))
        predictions = predictions.to(DEVICE)

        atten_list = []

        # y_in is initialized the first embedded char for each batch
        y_in = packed_targets[0]  # shape (b_size, emb_dim)

        sh = []
        sc = []

        # init states
        if mode == 'test' or mode == 'val':
            sh0 = init((1, o_size)).to(DEVICE)
            sh1 = init((1, o_size)).to(DEVICE)
            sc0 = init((1, o_size)).to(DEVICE)
            sc1 = init((1, o_size)).to(DEVICE)
            c0 = init((1, o_size)).to(DEVICE)
        else:
            sh0 = init((b_size, o_size)).to(DEVICE)
            sh1 = init((b_size, o_size)).to(DEVICE)
            sc0 = init((b_size, o_size)).to(DEVICE)
            sc1 = init((b_size, o_size)).to(DEVICE)
            c0 = init((b_size, o_size)).to(DEVICE)

        sh.append(sh0)
        sh.append(sh1)
        sc.append(sc0)
        sc.append(sc1)

        # forward at time -1 to train the hidden state
        _, c, sh, sc, _ = self.speller(hk, hv, y_in, c0, sh, sc, in_mask)

        pred = None
        
        # train mode
        if mode == 'train':
            # make predictions when we have train/val mode
            for idx in range(len(packed_targets) - 1):  # -1 because we do not use } as input
                # whether use teacher forcing
                # do not do teacher forcing at time 0
                if idx == 0:
                    tf_flag = False
                else:
                    tf_flag = (np.random.binomial(1, tf_rate, 1)[0] == 1)

                if tf_flag:
                    y_in = torch.argmax(pred, dim=1)
                    y_in = self.embedding(y_in)  # shape is (b_size, emb_dim)
                else:
                    y_in = packed_targets[idx]  # shape is (b_size, emb_dim)

                # take in target, c, sh, sc for this step and return pred, c, sh, sc for next step
                # pred should be compared with the target at next timestamp
                pred, c, sh, sc, atten_vec = self.speller(hk, hv, y_in, c, sh, sc, in_mask)
                predictions[:, idx, :] = pred

                atten_list.append(atten_vec)

            # transform attention list into torch tensor to be shown
            attention_heat_maps = torch.zeros((len(atten_list), len(atten_list[0])))
            for idx in range(len(atten_list)):
                attention_heat_maps[idx] = atten_list[idx]

            # y_targets are [1:] of each target
            return predictions, y_targets, attention_heat_maps

        # val/test mode, perform beam search
        else:
            searcher = PQ()
            pooler = PQ()

            sm = nn.Softmax(dim=1)

            end_symb = 33

            max_len = 150
            pred_len = 0
            person_id = 0

            states = (c, sh, sc)

            first_node = (0, person_id, append_char(None, 32), states)
            person_id += 1

            searcher.put(first_node)

            y_in = packed_targets[0] # shape (b_size, emb_dim)

            while pred_len <= max_len:
                new_searcher = PQ()
                temp_list = PQ()

                pred_len += 1

                # if pred_len > max_len:
                #     break

                # while searcher.qsize() > 0:

                #     parent = searcher.get()
                #     parent_prob = parent[0]  # negative log probability
                #     parent_path = parent[2]
                #     parent_c, parent_sh, parent_sc = parent[3]

                #     y_in = torch.zeros((1,), dtype=torch.long).to(DEVICE)
                #     y_in[0] = torch.tensor(parent_path[-1])
                #     y_in = self.embedding(y_in)

                    # child_probs, child_c, child_sh, child_sc, _ = self.speller(hk, hv, 
                    #     y_in, parent_c, parent_sh, parent_sc, in_mask)

                #     child_probs = sm(child_probs[0])

                #     for idx in range(num_letter):
                #         if child_probs[idx] > 0.001:
                #             child_prob = (parent_prob * (len(parent_path) ** beam_alpha) - 
                #                 torch.log(child_probs[idx])) / ((len(parent_path) + 1) ** beam_alpha)
                #             child_path = append_char(parent_path, idx)
                #             child_state = (child_c, child_sh, child_sc)
                #             child = (child_prob, person_id, child_path, child_state)
                #             person_id += 1

                #             temp_list.put(child)

                # get parents for this level
                parent_probs = []
                parent_paths = []
                parent_cs = []
                parent_shs = []
                parent_scs = []
                y_ins = []

                while searcher.qsize() > 0:
                    parent = searcher.get()
                    parent_prob = parent[0]  # negative log probability
                    parent_path = parent[2]
                    parent_c, parent_sh, parent_sc = parent[3]

                    y_in = torch.zeros((1,), dtype=torch.long).to(DEVICE)
                    y_in[0] = torch.tensor(parent_path[-1])
                    y_in = self.embedding(y_in)

                    # pack them into a batch
                    parent_probs.append(parent_prob)
                    parent_paths.append(parent_path)
                    parent_cs.append(parent_c)
                    parent_shs.append(parent_sh)
                    parent_scs.append(parent_sc)
                    y_ins.append(y_in)

                print("debug", y_in.shape, parent_c.shape, parent_sh[0].shape, parent_sc[0].shape)

                # process the batches into tensors
                beam_bsize = len(y_ins)

                print("beam_bsize:", beam_bsize)

                y_in_batch = torch.zeros((beam_bsize, embed_dim)).to(DEVICE)
                parent_c_batch = torch.zeros((beam_bsize, o_size)).to(DEVICE)
                parent_sh0_batch = torch.zeros((beam_bsize, o_size)).to(DEVICE)
                parent_sh1_batch = torch.zeros((beam_bsize, o_size)).to(DEVICE)
                parent_sc0_batch = torch.zeros((beam_bsize, o_size)).to(DEVICE)
                parent_sc1_batch = torch.zeros((beam_bsize, o_size)).to(DEVICE)

                for i in range(beam_bsize):
                    print("debug 3:", parent_sh1_batch.shape, parent_shs[i][0].shape)
                    y_in_batch[i] = y_ins[i]
                    parent_c_batch[i] = parent_cs[i]
                    parent_sh0_batch[i] = parent_shs[i][0]
                    parent_sh1_batch[i] = parent_shs[i][1]
                    parent_sc0_batch[i] = parent_scs[i][0]
                    parent_sc1_batch[i] = parent_scs[i][1]

                parent_sh_batch = (parent_sh0_batch, parent_sh1_batch)
                parent_sc_batch = (parent_sc0_batch, parent_sc1_batch)


                child_probs_batch, child_c_batch, child_sh_batch, child_sc_batch, _ = self.speller(hk, hv, y_in_batch,
                    parent_c_batch, parent_sh_batch, parent_sc_batch, in_mask)

                print("debug 1", y_in_batch.shape, parent_c_batch.shape, parent_sh_batch[0].shape, parent_sc_batch[0].shape)

                print("debug 2", child_probs_batch.shape, child_c_batch.shape, child_sh_batch[0].shape)

                child_probs_batch = sm(child_probs_batch)

                # generate all children
                for b in range(beam_bsize):
                    # get the parent states
                    parent_prob = parent_probs[b]  # negative log probability
                    parent_path = parent_paths[b]
                    parent_c = parent_c_batch[b]
                    parent_sh = parent_sh_batch[b]
                    parent_sc = parent_sc_batch[b]

                    child_probs = child_probs_batch[b]
                    child_c = child_c_batch[b]
                    child_sh = child_sh_batch[b]
                    child_sc = child_sc_batch[b]

                    # generate all children for that parent
                    for idx in range(num_letter):
                        if child_probs[idx] > 0.001:
                            child_prob = (parent_prob * (len(parent_path) ** beam_alpha) - 
                                torch.log(child_probs[idx])) / ((len(parent_path) + 1) ** beam_alpha)
                            child_path = append_char(parent_path, idx)
                            child_state = (child_c, child_sh, child_sc)
                            child = (child_prob, person_id, child_path, child_state)
                            person_id += 1

                            temp_list.put(child)

                while new_searcher.qsize() < beam_width and temp_list.qsize() > 0:
                    good_child = temp_list.get()
                    if good_child[2][-1] == end_symb:
                        pooler.put(good_child)
                    else:
                        new_searcher.put(good_child)

                searcher = new_searcher

            if pooler.qsize() == 0:
                best_path = append_char(None, 32)
                y_in = packed_targets[0]
                for idx in range(int(packed_targets.shape[0])):
                # for idx in range(200):
                    pred, c, sh, sc, atten_vec = self.speller(hk, hv, y_in, c, sh, sc, in_mask)
                    pred = sm(pred[0])
                    pred = torch.argmax(pred).cpu().numpy()
                    best_path = append_char(best_path, pred)

                    y_in = torch.zeros((1,), dtype=torch.long).to(DEVICE)
                    y_in[0] = torch.tensor(pred)
                    y_in = self.embedding(y_in)

            else:
                best_node = pooler.get()
                best_prob = best_node[0]
                best_path = best_node[2]

            return best_path


# the listener
class Listener(nn.Module):
    def __init__(self):
        super(Listener, self).__init__()
        global h_size
        global o_size
        self.rnn = nn.LSTM(input_size=40, hidden_size=h_size, num_layers=1, bidirectional=True)
        self.conv1 = nn.Conv1d(in_channels=2*h_size, out_channels=h_size, kernel_size=2, stride=2)
        self.rnn1 = nn.LSTM(input_size=h_size, hidden_size=h_size, num_layers=1, bidirectional=True)
        self.conv2 = nn.Conv1d(in_channels=2*h_size, out_channels=h_size, kernel_size=2, stride=2)
        self.rnn2 = nn.LSTM(input_size=h_size, hidden_size=h_size, num_layers=1, bidirectional=True)
        self.conv3 = nn.Conv1d(in_channels=2*h_size, out_channels=h_size, kernel_size=2, stride=2)
        self.rnn3 = nn.LSTM(input_size=h_size, hidden_size=h_size, num_layers=1, bidirectional=True)

        self.kLinear = nn.Linear(2*h_size, o_size)
        self.vLinear = nn.Linear(2*h_size, o_size)

        self.bn = nn.BatchNorm1d(2*h_size)

    def forward(self, utter_list):  # list
        # concat input on dim=0
        # packed_input[0] has shape (sum(l), 40), dtype=float
        # packed_input[1] has shape (max(l)), dtype=int
        packed_input = rnn.pack_sequence(utter_list)

        # packed_output[0] has shape (sum(l), hidden_size or *2), dtype=float
        # packed_output[1] has shape (max(l)), dtype=int
        packed_output, _ = self.rnn(packed_input, None)

        # shape (max(l), batch_size, hidden_size or *2), dtype=float
        # padded_output[:,i,:] (shape is (max(l), hidden_size or *2)) corresponds to the i'th input
        # lens are lengths for each tensor before padding(after padding will all be max(l))
        padded_output, lens = rnn.pad_packed_sequence(packed_output)

        # go through pBLSTM
        padded_output = trans(padded_output, True)
        padded_output = self.conv1(padded_output)
        # padded_output = self.bn(padded_output)
        padded_output = trans(padded_output, False)
        padded_output, _ = self.rnn1(padded_output)

        padded_output = trans(padded_output, True)
        padded_output = self.conv2(padded_output)
        # padded_output = self.bn(padded_output)
        padded_output = trans(padded_output, False)
        padded_output, _ = self.rnn2(padded_output)

        padded_output = trans(padded_output, True)
        padded_output = self.conv3(padded_output)
        # padded_output = self.bn(padded_output)
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

        self.sm = nn.Softmax(dim=1)

        self.linear1 = nn.Linear(o_size, o_size)
        self.linear2 = nn.Linear(o_size, o_size)
        self.activate = nn.ReLU()
        self.linear3 = nn.Linear(o_size, num_letter)

        self.bn = nn.BatchNorm1d(h_size)

    # _1 means params from time t-1
    # return values are all for time t
    def forward(self, hk, hv, y_1, c_1, sh_1, sc_1, mask):
        global o_size

        sh = []
        sc = []

        # calculate state for t
        sh0, sc0 = self.rnnCell0(torch.cat((y_1, c_1), dim=1), (sh_1[0], sc_1[0]))
        sh1, sc1 = self.rnnCell1(sh0, (sh_1[1], sc_1[1]))
        # sh1 = self.bn(sh1)
        # sc1 = self.bn(sc1)

        sh.append(sh0)
        sh.append(sh1)
        sc.append(sc0)
        sc.append(sc1)

        # calculate attention
        temp = torch.bmm(hk, sh1.reshape(sh1.size(0), sh1.size(1), 1))
        temp = temp.reshape(temp.size(0), temp.size(1))
        temp = self.sm(temp)

        temp = temp * mask
        temp /= torch.sum(temp, dim=1).reshape(-1, 1)

        atten_vec = temp[0]  # take the attention vec of sample 0 in the batch for debug

        attention = temp.reshape(temp.size(0), 1, temp.size(1))

        # get the context
        c = torch.bmm(attention, hv)
        c = c.reshape((c.size(0), c.size(2)))

        # calculate prediction
        out = self.linear1(sh1)
        out += self.linear2(c)
        out = self.bn(out)
        out = self.activate(out)
        out = self.linear3(out)
        return out, c, sh, sc, atten_vec


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


def get_mask(h_lens, mode):
    global o_size
    global b_size

    if mode == 'test' or mode == 'val':
        mask = torch.zeros((1, h_lens[0]))
        mask[0][:h_lens[0]] = 1
    else:
        mask = torch.zeros((b_size, h_lens[0]))
        for i in range(b_size):
            mask[i][:h_lens[i]] = 1
    
    return mask


# append the input char to the list
def append_char(char_list, char):
    if not char_list is None:
        char_list = np.append(char_list, char)
    else:
        char_list = [char]
    return char_list


if __name__ == '__main__':
    pass
