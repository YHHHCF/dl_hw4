import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import numpy as np
from dataloader import *
from model import *
from Levenshtein import distance


if __name__ == '__main__':
    val_loader = get_loader('val', 4)

    model = LAS()

    for inputs, targets in val_loader:
        out = model(inputs, targets)

        print(out.shape)

        break

