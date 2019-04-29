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

    cnt = 0

    max_len = 200
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            char = [[32, 33]]  # the init target
            for i in range(max_len):
                predictions = model(inputs, char)
                print(predictions.shape)
                pred = predictions[i][:len(target)]

                pred_str = toSentence(torch.argmax(pred, dim=1))
                target_str = toSentence(target)
                dis += distance(pred_str, target_str) / len(pred_str)

            total_loss += (loss / b_size)
            total_distance += (dis / b_size)
            cnt += 1

    total_loss /= cnt
    total_distance /= cnt
    
    print("Val loss: {}, distance: {}".format(total_loss, total_distance))
    writer.add_scalar('val/loss', total_loss, ep)
    writer.add_scalar('val/distance', total_distance, ep)
    return total_distance


if __name__ == '__main__':
    b_size = 1

    test_loader = get_loader('test', b_size)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    path = './../result/model_exp1.t7'
    model, _ = load_ckpt(path, 'test')
    model = model.to(DEVICE)

    test(model, test_loader)


