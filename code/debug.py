from dataloader import *

train_loader = get_loader('train', 4)
val_loader = get_loader('val', 4)
test_loader = get_loader('test', 4)

for idx, (data, label) in enumerate(train_loader):
    for d in data:
        print("train data shape", d.shape)
    for l in label:
        print("train label shape", l.shape)
    break

for idx, (data, label) in enumerate(val_loader):
    for d in data:
        print("val data shape", d.shape)
    for l in label:
        print("val label shape", l.shape)
    break

for idx, data in enumerate(test_loader):
    for d in data:
        print("test data shape", d.shape)
    break
