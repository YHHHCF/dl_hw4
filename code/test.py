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
    model.eval()

    max_len = 200
    
    with torch.no_grad():

        for inputs, targets in test_loader:
            char = []  # the init target
            char.append(torch.tensor([32, 33]).to(DEVICE))
            preds = []

            for i in range(max_len):
                pred = model(inputs, char, 'prediction')
                pred = pred.reshape(pred.shape[2])
                char = []
                pred = torch.argmax(pred)

                char.append(torch.tensor([pred, 33]).to(DEVICE))
                preds.append(pred)

            pred_str = toSentence(preds)
            print(pred_str)
    return None


if __name__ == '__main__':
    b_size = 1

    test_loader = get_loader('test', b_size)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    path = './../result/model_exp1.t7'
    model, _ = load_ckpt(path, 'test')
    model = model.to(DEVICE)

    test(model, test_loader)

