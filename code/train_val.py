from dataloader import *
from model import *
from label_proc import *
from Levenshtein import distance
from tensorboardX import SummaryWriter
from config import *


def train(epochs, train_loader, val_loader, model, optim, sche, writer):
    global tf_rate
    model.train()

    idx = 0
    for e in range(epochs):
        print("begin epoch {}  ===============".format(e))
        print("teacher forcing rate is", tf_rate)
        for inputs, targets in train_loader:
            optim.zero_grad()
            predictions, atten_heatmap = model(inputs, targets)

            # calculate loss and distance and backprop
            loss = 0.
            dis = 0.
            for i in range(len(predictions)):
                target = targets[i][1:-1]
                pred = predictions[i][:-1]
                loss += torch.exp(criterion(pred, target))

                pred_str = toSentence(torch.argmax(pred, dim=1))
                target_str = toSentence(target)
                dis += distance(pred_str, target_str)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_thresh)
            optim.step()

            print("Step {} loss: {}, distance: {}".format(idx, (loss / b_size), (dis / b_size)))

            writer.add_scalar('train/loss', (loss / b_size), idx)
            writer.add_scalar('train/distance', (dis / b_size), idx)

            if idx % 20 == 0:
                atten_heatmap = atten_heatmap.detach().numpy()
                atten_heatmap = atten_heatmap.reshape(1, atten_heatmap.shape[0], atten_heatmap.shape[1])
                writer.add_image('atten_heatmap' + str(idx), atten_heatmap, idx)
            idx += 1

        val_loss, val_dis = val(model, val_loader, writer, e)

        if val_loss < best_loss:
            save_ckpt(model, optim, val_loss, e)
            print("A model is saved!")
            best_loss = val_loss

        sche.step()

        save_ckpt(model, optim, val_loss, e)
        model.train()

        if tf_rate < tf_thresh and e >= begin_tf:
            tf_rate += tf_incr
        
    return


# validation
def val(model, val_loader, writer, ep):
    model.eval()

    total_dis = 0.
    total_loss = 0.
    cnt = 0.
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            predictions = model(inputs, targets, 'val')

            # calculate loss and distance
            loss = 0.
            dis = 0.

            # go through each item in the batch
            for idx in range(len(predictions)):
                target = targets[idx][1:-1]
                pred = predictions[idx][:-1]

                pred_str = toSentence(pred)
                target_str = toSentence(target)
                dis += distance(pred_str, target_str)

                if len(target) > len(pred):
                    pad = torch.full((len(target) - len(pred)), eos)
                    pred = torch.cat((pred, pad))

                elif len(pred) > len(target):
                    pad = torch.full((len(pred) - len(target)), eos)
                    target = torch.cat((target, pad))

                loss += criterion(pred, target)

            total_dis += (dis / len(predictions))
            total_loss += (loss / len(predictions))
            cnt += 1

    total_dis /= cnt
    total_loss /= cnt
    
    print("Val loss: {}; Val distance: {}".format(total_loss, total_dis))
    writer.add_scalar('val/loss', total_loss, ep)
    writer.add_scalar('val.dis', total_dis, ep)
    return total_loss, total_dis


def save_ckpt(model, optim, val_loss, e):
    global exp_id
    path = './../result/model_exp' + str(exp_id) + '_' + str(e) + '.t7'

    torch.save({
        'val_loss': val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
    }, path)
    return


def load_ckpt(path, mode='train'):
    global best_loss
    new_model = LAS()
    pretrained_ckpt = torch.load(path)
    new_model.load_state_dict(pretrained_ckpt['model_state_dict'])

    print("loaded a pretrained model with loss:", pretrained_ckpt['val_loss'])
    
    if mode == 'train':

        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=lr, weight_decay=wd)
        new_optimizer.load_state_dict(pretrained_ckpt['optimizer_state_dict'])

        for state in new_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)

        if pretrained_ckpt['val_loss'] < best_loss:
            best_loss = pretrained_ckpt['val_loss']

        return new_model, new_optimizer

    else:
        return new_model


if __name__ == '__main__':
    train_loader = get_loader('train', b_size)
    val_loader = get_loader('val', b_size)

    if if_pretrain:
        model, optimizer = load_ckpt(pretrain_path, 'train')
    else:
        model = LAS()
        model.apply(init_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(DEVICE)

    writer = SummaryWriter()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    train(epochs, train_loader, val_loader, model, optimizer, scheduler, writer)
