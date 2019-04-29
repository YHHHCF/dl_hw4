import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import numpy as np
from dataloader import *
from model import *
from label_proc import *
from Levenshtein import distance
from tensorboardX import SummaryWriter


def train(epochs, train_loader, val_loader, model, optim, writer):
    global b_size
    model.train()

    idx = 0
    for e in range(epochs):
        print("begin epoch {}  ===============".format(e))
        for inputs, targets in train_loader:
            optim.zero_grad()
            predictions, y_targets = model(inputs, targets)

            # calculate loss and distance and backprop
            loss = 0.
            dis = 0.
            for i in range(len(y_targets)):
                target = y_targets[i]
                pred = predictions[i][:len(target)]
                loss += torch.exp(criterion(pred, target))

                pred_str = toSentence(torch.argmax(pred, dim=1))
                target_str = toSentence(target)
                dis += distance(pred_str, target_str) / len(pred_str)

            loss.backward()
            optim.step()

            print("loss: {}, distance: {}".format((loss / b_size), (dis / b_size)))

            writer.add_scalar('train/loss', (loss / b_size), idx)
            writer.add_scalar('train/distance', (dis / b_size), idx)
            idx += 1

        val_dis = val(model, val_loader, writer, e)
        model.train()

        if val_dis < best_dis:
            save_ckpt(model, optim, val_dis)
            print("A model is saved!")
    return


# validation
def val(model, val_loader, writer, ep):
    global b_size
    model.eval()

    total_loss = 0.
    total_distance = 0.
    cnt = 0.
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            predictions, y_targets = model(inputs, targets)

            # calculate loss and distance and backprop
            loss = 0.
            dis = 0.
            for i in range(len(y_targets)):
                target = y_targets[i]
                pred = predictions[i][:len(target)]
                loss += torch.exp(criterion(pred, target))

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


def save_ckpt(model, optim, val_dis):
    path = './../result/model_exp1.t7'

    torch.save({
        'val_dis': val_dis,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
    }, path)
    return


def load_ckpt(path):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    new_model = LAS()
    pretrained_ckpt = torch.load(path)
    new_model.load_state_dict(pretrained_ckpt['model_state_dict'])
    
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
    new_optimizer.load_state_dict(pretrained_ckpt['optimizer_state_dict'])

    for state in new_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(DEVICE)

    print("loaded a pretrained model with distance:", pretrained_ckpt['val_dis'])

    return new_model, new_optimizer


if __name__ == '__main__':
    b_size = 256
    epochs = 20
    best_dis = 1

    train_loader = get_loader('train', b_size)
    val_loader = get_loader('val', b_size)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = LAS()
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    writer = SummaryWriter()

    train(epochs, train_loader, val_loader, model, optimizer, writer)

