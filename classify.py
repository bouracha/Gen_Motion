import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, classification_report

import models.classifier as nnmodel
import utils.supervised_data as supervised_data

from opt import Options
opt = Options().parse()

l2_reg = 0.0
if opt.l2_reg:
    l2_reg = 1e-4

folder_name=opt.name
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = "cuda"
else:
    device = "cpu"

# ===============================================================
#                     Data
# ===============================================================
train_batch_size=opt.train_batch_size
test_batch_size=opt.test_batch_size
if opt.use_MNIST:
    ### MNIST
    folder_name = folder_name+"MNIST"
    from torchvision import transforms
    from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        MNIST('.', train=True, download=False, transform=transform),
        batch_size=train_batch_size,
        shuffle=True)
    val_loader = DataLoader(
        MNIST('.', train=False, download=False, transform=transform),
        batch_size=train_batch_size,
        shuffle=True)
    input_n=784
    output_n=10
else:
    train_dataset = supervised_data.ClassifierDataset(str(opt.data_path)+"train.csv")
    val_dataset = supervised_data.ClassifierDataset(str(opt.data_path)+"val.csv")
    test_dataset = supervised_data.ClassifierDataset(str(opt.data_path)+"test.csv")

    train_batch_size=opt.train_batch_size
    test_batch_size=opt.test_batch_size
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.train_batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=test_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)
    input_n = train_dataset.n
    output_n = train_dataset.num_classes
# ===============================================================
# Instantiate model, and methods used fro training and valdation
# ===============================================================

model = nnmodel.Classifier(num_features=input_n, hidden_layers=opt.hidden_layers, num_classes=output_n , device=device, act_fn=nn.LeakyReLU(0.1), batch_norm=opt.batch_norm, p_dropout=opt.p_drop)
model._initialise(start_epoch=opt.start_epoch, folder_name=folder_name, lr=opt.lr, l2_reg=l2_reg, train_batch_size=train_batch_size)


if not opt.inference:
    for epoch in range(opt.start_epoch, opt.n_epochs + 1):
        print("Epoch: {}/{}".format(epoch, opt.n_epochs))

        model.train_epoch(epoch, train_loader)
        model.eval_full_batch(train_loader, epoch, 'train')
        model.eval_full_batch(val_loader, epoch, 'val')

        model.save_checkpoint_and_csv(epoch)


if opt.inference:
    y_pred_list = []
    y_gt_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, y_gt in val_loader:
            cur_batch_size = len(X_batch)
            X_batch = X_batch.reshape(cur_batch_size, -1)

            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)

            y_pred_softmax = torch.log_softmax(y_test_pred, dim = 1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

            if y_pred_list == []:
                y_pred_list = y_pred_tags.cpu().numpy()
                y_gt_list = y_gt.cpu().numpy()
            else:
                y_pred_list = np.concatenate((y_pred_list, y_pred_tags.cpu().numpy()))
                y_gt_list = np.concatenate((y_gt_list, y_gt.cpu().numpy()))

    plt.figure(figsize=(20,15))
    fontsize=30
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_gt_list, y_pred_list)).rename(columns=train_dataset.idx2class, index=train_dataset.idx2class)
    heatmap_plot = sns.heatmap(confusion_matrix_df/(1.0*np.sum(confusion_matrix_df)), annot=True, fmt='.2%')
    plt.xlabel('Predicted', fontsize=fontsize)
    plt.ylabel('Actual', fontsize=fontsize)
    plt.savefig(model.folder_name+"/confusion.png")

    print(classification_report(y_gt_list, y_pred_list))