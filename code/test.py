import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import numpy as np
from dataloader import *
from model import *
from label_proc import *
from train_val import *
from Levenshtein import distance
import os, sys

answer_path = './answer.npz'

# for debug
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


# predict without label on val dataset
def test_val(model, val_loader, writer):
    print("test with batch size: 1")
    char = 32
    end_symb = 33
    first_node = (-1, append_char(None, char))

    model.eval()

    cnt = 0
    total_dis = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            print("================")
            print("term:", cnt)

            # debug
            # if not cnt == 40:
            #     cnt += 1
            #     continue

            label_str = toSentence(targets[0])
            print("label:", label_str[1:-1])
                
            prediction = model(inputs, targets, 'test')

            pred_str = toSentence(prediction)
            
            print("pred sentence:", pred_str[1:-1])
            
            dis = distance(pred_str, label_str)

            print("label/pred length: {}, {}; distance: {}".format(len(label_str), len(pred_str), dis))

            # writer.add_scalar('test/distance', dis, cnt)

            total_dis += dis
            cnt += 1

            if cnt == 100:
                break
            
    print("total_dis", total_dis / cnt)
    return None

def test(model, test_loader):
    global answer_path

    # badcase = [14, 72, 111, 212, 300, 448, 499]

    char = 32
    end_symb = 33
    first_node = (-1, append_char(None, char))

    model.eval()

    idx = 0

    answer_dict = {}
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            # if idx in badcase:
            print("================")
            print("term:", idx)
            prediction = model(inputs, targets, 'test')

            pred_str = toSentence(prediction)

            pred_str = pred_str[1:-1]
            
            # print("pred sentence:", pred_str)

            answer_dict[idx] = pred_str

            idx += 1

        np.savez(answer_path, answer_dict)       

    return


if __name__ == '__main__':

    val_loader = get_loader('test_val', batch_size=1) # use the val dataset to perform beam search
    test_loader = get_loader('test', batch_size=1) # use the test dataset to perform beam search

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    path = './../result/model_exp7_19.t7'
    model, _ = load_ckpt(path, 'test')  
    model = model.to(DEVICE)

    writer = SummaryWriter()

    test(model, test_loader)

    # ckpts = [19]
    # for i in ckpts:
    #     path = './../result/model_exp7_' + str(i) + '.t7'
    #     print(path)
    #     model, _ = load_ckpt(path, 'test')  
    #     model = model.to(DEVICE)
    #     writer = SummaryWriter()

    #     test_val(model, val_loader, writer)        

