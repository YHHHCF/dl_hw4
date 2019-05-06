from label_proc import *
from queue import PriorityQueue as PQ
from model_helper import *


class LAS(nn.Module):
    def __init__(self):
        super(LAS, self).__init__()

        self.listener = Listener()
        self.speller = Speller()

        self.embedding = nn.Embedding(num_letter, embed_dim)
        # self.embedding = nn.Linear(num_letter, embed_dim)

        self.sm0 = nn.Softmax(dim=0)
        self.sm1 = nn.Softmax(dim=0)

    # forward is for train mode
    def forward(self, utter_list, targets):
        hk, hv, h_lens = self.listener(utter_list)

        emb_targets = []
        for t in targets:
            # t = get_emb(t)
            t = self.embedding(t)  # shape (L, E)
            emb_targets.append(t)
        packed_targets = rnn.pad_sequence(emb_targets)  # shape (L, B, E)

        # a mask of shape(B, L) to mask input attention
        in_mask = get_mask(h_lens)

        # predictions shape (B, L - 1, V)
        predictions = torch.zeros((b_size, len(packed_targets) - 1, num_letter)).to(DEVICE)

        atten_list = []

        # y_in is initialized the first embedded char for each batch
        y_in = packed_targets[0]  # shape (B, E)

        sh = []
        sc = []

        # init states and context, all have shape (B, O)
        sh0 = zero_init((b_size, o_size))
        sh1 = zero_init((b_size, o_size))
        sc0 = zero_init((b_size, o_size))
        sc1 = zero_init((b_size, o_size))
        c0 = zero_init((b_size, o_size))

        sh.append(sh0)
        sh.append(sh1)
        sc.append(sc0)
        sc.append(sc1)

        # forward at time -1 to train the hidden state
        _, c, sh, sc, _ = self.speller(hk, hv, y_in, c0, sh, sc, in_mask)

        pred = None
        
        # iterate under train mode
        for idx in range(len(packed_targets) - 1):  # -1 because we do not use } as input
            # whether use teacher forcing
            # do not do teacher forcing at time 0
            if idx == 0:
                tf_flag = False
            else:
                tf_flag = (np.random.binomial(1, tf_rate, 1)[0] == 1)

            if tf_flag:
                y_in = torch.argmax(pred, dim=1)  # shape (B, 1)
                y_in = self.embedding(y_in)  # shape (B, E)
            else:
                y_in = packed_targets[idx]  # shape (B, E)

            # take in target, c, sh, sc for this step and return pred, c, sh, sc for next step
            # pred shape (B, V)
            pred, c, sh, sc, atten_vec = self.speller(hk, hv, y_in, c, sh, sc, in_mask)
            predictions[:, idx, :] = pred

            atten_list.append(atten_vec)

        # attention_heat_maps shape (L_target - 1, L_h)
        attention_heat_maps = torch.zeros((len(atten_list), len(atten_list[0])))
        for idx in range(len(atten_list)):
            attention_heat_maps[idx] = atten_list[idx]

        return predictions, attention_heat_maps

    # predict is for val and test mode
    def predict(self, utter_list, mode):
        global max_len
        batch_size = len(utter_list)
        node_id = 0  # make each node unique

        if mode == 'val':
            beam_width = beam_width_val
        else:
            beam_width = beam_width_test

        searchers = []
        poolers = []

        for b in range(batch_size):
            searchers.append(PQ())
            poolers.append(PQ())

        # hk, hv shape (B, L, O)
        hk, hv, h_lens = self.listener(utter_list)

        # a mask of shape(B, L) to mask input attention
        in_mask = get_mask(h_lens)

        # y_in is initialized the first embedded char for each batch
        y_in = torch.full((batch_size, 1), sos)  # shape (B, 1)
        y_in = self.embedding(y_in)  # shape (B, E)

        # init states and context
        sh = []
        sc = []

        sh0 = zero_init((batch_size, o_size))
        sh1 = zero_init((batch_size, o_size))
        sc0 = zero_init((batch_size, o_size))
        sc1 = zero_init((batch_size, o_size))
        c0 = zero_init((batch_size, o_size))

        sh.append(sh0)
        sh.append(sh1)
        sc.append(sc0)
        sc.append(sc1)

        # forward at time -1, c, sh[0/1], sc[0/1] shape (B, O)
        _, c, sh, sc, _ = self.speller(hk, hv, y_in, c0, sh, sc, in_mask)

        # iterate under val/test mode
        states = (c, sh, sc)

        for b in range(batch_size):
            first_node = (0, node_id, append_char(None, sos), states)
            node_id += 1
            searchers[b].put(first_node)

        for l in range(max_len):
            if l == 0:
                bw = 1  # the number of parent nodes in each batch
            else:
                bw = beam_width

            temp_pq = []  # PQ for each batch to store all the temp nodes
            for b in range(batch_size):
                temp_pq.append(PQ())

            # go through each node in each batch
            for iter in range(bw):
                # init all the params
                y_in = zero_init(batch_size)
                sh0 = zero_init((batch_size, o_size))
                sh1 = zero_init((batch_size, o_size))
                sc0 = zero_init((batch_size, o_size))
                sc1 = zero_init((batch_size, o_size))
                c = zero_init((batch_size, o_size))

                parent_probs = zero_init(batch_size)
                parent_paths = []

                sh = []
                sc = []

                # combine the states before going to speller
                for b in range(batch_size):
                    parent = searchers[b].get()

                    parent_prob = parent[0]
                    parent_path = parent[2]
                    parent_c, parent_sh, parent_sc = parent[3]

                    y_in[b] = parent_path[-1]
                    sh0[b] = parent_sh[0]
                    sh1[b] = parent_sh[1]
                    sc0[b] = parent_sc[0]
                    sc1[b] = parent_sc[1]
                    c[b] = parent_c

                    parent_probs[b] = parent_prob
                    parent_paths.append(parent_path)

                y_in_emb = self.embedding(y_in)  # shape (B, E)
                sh.append(sh0)
                sh.append(sh1)
                sc.append(sc0)
                sc.append(sc1)

                # go through speller in a batch
                pred, c, sh, sc, _ = self.speller(hk, hv, y_in_emb, c, sh, sc, in_mask)

                # extract and split predictions and states
                probs = self.sm1(pred)  # a batch of child prob of shape (B, V)
                sh0s = sh[0]  # shape (B, O)
                sh1s = sh[1]  # shape (B, O)
                sc0s = sc[0]  # shape (B, O)
                sc1s = sc[1]  # shape (B, O)

                # get the results into the temp queue
                for b in range(batch_size):
                    for idx in range(num_letter):
                        if idx == sos:  # do not consider sos
                            continue
                        else:
                            child_prob = (parent_probs[b] * (len(parent_paths[b]) ** beam_alpha) -
                                          torch.log(probs[b][idx])) / ((len(parent_paths[b]) + 1) ** beam_alpha)

                            sh_child = []
                            sc_child = []

                            sh_child.append(sh0s[b])
                            sh_child.append(sh1s[b])
                            sc_child.append(sc0s[b])
                            sc_child.append(sc1s[b])

                            states = (c[b], sh_child, sc_child)

                            child = (child_prob, node_id, append_char(parent_paths[b], idx), states)
                            node_id += 1

                            temp_pq[b].put(child)

            # put top results in temp to searchers or poolers
            searchers = []  # init a new searcher for the next length
            for b in range(batch_size):
                if l == max_len - 1:
                    node = temp_pq[b].get()
                    poolers[b].put(node)

                else:
                    searcher = PQ()
                    while searcher.qsize() < beam_width:
                        node = temp_pq[b].get()
                        if node[2][-1] == eos:
                            poolers[b].put(node)
                        else:
                            searcher.put(node)

                    searchers.append(searcher)

        best_paths = []
        for b in range(b):
            good_node = poolers[b].get()
            path = good_node[2]
            best_paths.append(path)

        return best_paths


if __name__ == '__main__':
    pass
