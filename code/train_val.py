import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import numpy as np
from dataloader import *
from model import *
from label_proc import *
from Levenshtein import distance


if __name__ == '__main__':
    b_size = 64
    epoch = 20
    val_loader = get_loader('train', b_size)

    model = LAS()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for e in range(epoch):
        for inputs, targets in val_loader:
            optimizer.zero_grad()
            predictions, y_targets = model(inputs, targets)

            # calculate loss and distance and backprop
            loss = 0
            dis = 0.
            for i in range(len(y_targets)):
                target = y_targets[i]
                pred = predictions[i][:len(target)]
                loss += criterion(pred, target)

                pred_str = toSentence(torch.argmax(pred, dim=1))
                target_str = toSentence(target)
                dis += distance(pred_str, target_str) / len(pred_str)

            print("loss: {}, distance: {}".format((loss / 64), (dis / 64)))
            loss.backward()
            optimizer.step()
