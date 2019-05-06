from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import *

train_data_path = '../data/train.npy'
train_label_path = '../data/train_label_int.npy'

val_data_path = '../data/dev.npy'
val_label_path = '../data/val_label_int.npy'

test_data_path = '../data/test.npy'


# utterance dataset, init dataset with data and label
class UtterDataset(Dataset):
    def __init__(self, utters, labels, mode):
        self.data = [torch.tensor(utter) for utter in utters]

        if not mode == 'test':
            self.labels = [torch.tensor(label) for label in labels]

        self.mode = mode
        print("init dataset which contains {} data".format(len(self.data)))

    def __getitem__(self, i):
        data = self.data[i]
        data = data.type(torch.float32)

        if not self.mode == 'test':
            label = self.labels[i]
            return data.to(DEVICE), label.to(DEVICE)
        else:
            return data.to(DEVICE)

    def __len__(self):
        return len(self.data)


# collate_utter return your data sorted by length
def collate_utter(utter_list):
    if len(utter_list[0]) == 2: # a tuple, so in train/val mode
        inputs, targets = zip(*utter_list)
        lens = [len(utter) for utter in inputs]
        seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
        inputs = [inputs[i] for i in seq_order]
        targets = [targets[i] for i in seq_order]
        return inputs, targets
    else: # a tensor, so in test mode
        inputs = utter_list
        lens = [len(utter) for utter in inputs]
        seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
        inputs = [inputs[i] for i in seq_order]
        return inputs


# get the dataloader, mode can be train/val/test
def get_loader(mode, batch_size):
    if mode == 'train':
        data_path = train_data_path
        label_path = train_label_path
        sf = True
        drop = True
    elif mode == 'val':
        data_path = val_data_path
        label_path = val_label_path
        sf = False
        drop = False
    else:  # test mode
        data_path = test_data_path
        sf = False
        drop = False

    data = np.load(data_path, allow_pickle=True, encoding='bytes')

    if mode == 'test':
        label = None
    else:
        label = np.load(label_path, allow_pickle=True, encoding='bytes')

    dataset = UtterDataset(data, label, mode)
    loader = DataLoader(dataset, shuffle=sf, batch_size=batch_size, drop_last=drop, collate_fn=collate_utter)

    return loader


if __name__ == '__main__':
    pass
