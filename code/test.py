import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import numpy as np
from dataloader import *
from model import *
from label_proc import *
from train_val import *


# test without label
def test(model, test_loader):
    global b_size
    char = 32
    end_symb = 33
    first_node = (-1, append_char(None, char))

    model.eval()

    max_len = 200
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            print("debug 1:", len(inputs))
            prediction = model(inputs, targets, 'test')
            print(prediction)


    return None


if __name__ == '__main__':
    global b_size

    val_loader = get_loader('test', b_size)
    test_loader = get_loader('test', b_size)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    path = './../result/model_exp5_27.t7'
    model, _ = load_ckpt(path, 'test')
    model = model.to(DEVICE)

    test(model, test_loader)

