import torch

# model config
b_size = 256
h_size = 128
o_size = 128
num_letter = 34
embed_dim = 256

# beam search config
beam_width_val = 8
beam_width_test = 32
beam_alpha = 0.7
max_len = 150

# for training
epochs = 40
best_loss = 10
lr = 1e-3
wd = 1e-6

clip_thresh = 1000

tf_rate = 0.0
begin_tf = 10
tf_thresh = 0.2
tf_incr = 0.01

# for scheduler
step_size = 10
gamma = 0.3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# data pre-process
sos = 32
eos = 33

# pretrain settings
if_pretrain = False
pretrain_path = './../result/model_exp8_9.t7'

exp_id = 9

answer_path = './answer.npz'
