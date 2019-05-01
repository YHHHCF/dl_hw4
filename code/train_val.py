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
    global tf_rate
    global best_dis

    model.train()

    idx = 0
    for e in range(epochs):
        print("begin epoch {}  ===============".format(e))
        print("teacher forcing rate is", tf_rate)
        for inputs, targets in train_loader:
            optim.zero_grad()
            predictions, y_targets, atten_heatmap = model(inputs, targets, 'train')

            # calculate loss and distance and backprop
            loss = 0.
            dis = 0.
            for i in range(len(y_targets)):
                target = y_targets[i]
                pred = predictions[i][:len(target)]
                loss += torch.exp(criterion(pred, target))

                pred_str = toSentence(torch.argmax(pred, dim=1))
                target_str = toSentence(target)
                dis += distance(pred_str, target_str)

                # if i == 0:
                #     print("pred:", pred_str)
                #     print("target:", target_str)
                #     print("len:", len(pred_str), len(target_str))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optim.step()

            print("Step {} loss: {}, distance: {}".format(idx, (loss / b_size), (dis / b_size)))

            writer.add_scalar('train/loss', (loss / b_size), idx)
            writer.add_scalar('train/distance', (dis / b_size), idx)

            if idx % 20 == 0:
                atten_heatmap = atten_heatmap.detach().numpy()
                atten_heatmap = atten_heatmap.reshape(1, atten_heatmap.shape[0], atten_heatmap.shape[1])
                writer.add_image('atten_heatmap' + str(idx), atten_heatmap, idx)
            idx += 1


        val_dis = val(model, val_loader, writer, e)
        model.train()

        tf_rate += 0.01

        if val_dis < best_dis:
            save_ckpt(model, optim, val_dis)
            print("A model is saved!")
            best_dis = val_dis
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
            predictions, y_targets, _ = model(inputs, targets, 'train')

            # calculate loss and distance and backprop
            loss = 0.
            dis = 0.
            for i in range(len(y_targets)):
                target = y_targets[i]
                pred = predictions[i][:len(target)]
                loss += torch.exp(criterion(pred, target))

                pred_str = toSentence(torch.argmax(pred, dim=1))
                target_str = toSentence(target)
                dis += distance(pred_str, target_str)

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
    path = './../result/model_exp3.t7'

    torch.save({
        'val_dis': val_dis,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
    }, path)
    return


def load_ckpt(path, mode='train'):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    new_model = LAS()
    pretrained_ckpt = torch.load(path)
    new_model.load_state_dict(pretrained_ckpt['model_state_dict'])
    
    if mode == 'train':
        global lr
        global best_dis

        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=lr)
        new_optimizer.load_state_dict(pretrained_ckpt['optimizer_state_dict'])

        for state in new_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)

        if pretrained_ckpt['val_dis'] < best_dis:
            best_dis = pretrained_ckpt['val_dis']

    else:
        new_optimizer = None

    print("loaded a pretrained model with distance:", pretrained_ckpt['val_dis'])

    return new_model, new_optimizer


if __name__ == '__main__':
    epochs = 100
    best_dis = 100
    lr = 1e-2

    if_pretrain = False
    path = './../result/model_exp3.t7'

    train_loader = get_loader('train', b_size)
    val_loader = get_loader('val', b_size)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if if_pretrain:
        model, optimizer = load_ckpt(path, 'train')
    else:
        model = LAS()
        model.apply(init_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(DEVICE)

    writer = SummaryWriter()

    train(epochs, train_loader, val_loader, model, optimizer, writer)

