import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np

train_data_path = '../data/train.npy'
train_label_path = '../data/train_label_int.npy'

val_data_path = '../data/dev.npy'
val_label_path = '../data/val_label_int.npy'

test_data_path = '../data/test.npy'


# utterance dataset, init dataset with data and label
class UtterDataset(Dataset):
    def __init__(self, utters, labels):
        self.data = [torch.tensor(utter) for utter in utters]
        self.labels = [torch.tensor(label) for label in labels]
        print("init dataset with {} data and {} label".format(len(self.data), len(self.labels)))

    def __getitem__(self, i):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        data = self.data[i]
        label = self.labels[i]
        data = data.type(torch.float32)
        return data.to(DEVICE), label.to(DEVICE)

    def __len__(self):
        return len(self.labels)


# collate_utter return your data sorted by length
def collate_utter(utter_list):
    inputs, targets = zip(*utter_list)
    lens = [len(utter) for utter in inputs]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    inputs = [inputs[i] for i in seq_order]
    targets = [targets[i] for i in seq_order]
    return inputs, targets


# get the dataloader
def get_loader(mode, batch_size):
    if mode == 'train':
        data_path = train_data_path
        label_path = train_label_path
        sf = True
    elif mode == 'val':
        data_path = val_data_path
        label_path = val_label_path
        sf = False
    else:
        data_path = test_data_path
        label_path = val_label_path
        sf = False

    data = np.load(data_path, allow_pickle=True, encoding='bytes')
    label = np.load(label_path, allow_pickle=True, encoding='bytes')

    if mode == 'test':
        label = label[:len(data)]
        batch_size = 1

    dataset = UtterDataset(data, label)
    loader = DataLoader(dataset, shuffle=sf, batch_size=batch_size, drop_last=True, 
        collate_fn=collate_utter)

    return loader


if __name__ == '__main__':
    pass
