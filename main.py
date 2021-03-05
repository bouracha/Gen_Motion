from __future__ import print_function, absolute_import, division

import torch.optim

import data as data

import models.VAE as nnmodel

from opt import Options
opt = Options().parse()

weight_decay = 0.0
if opt.weight_decay:
    weight_decay = 1e-4

folder_name=opt.name
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = "cuda"
    job = 10 #Can change this to multi-process dataloading
else:
    device = "cpu"
    job = 0

# ===============================================================
# Load data
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
else:
    ### Human Motion Data
    data = data.DATA("h3.6m_3d", "h3.6m/dataset/")
    out_of_distribution = data.get_poses(input_n=1, output_n=1, sample_rate=2, dct_n=2, out_of_distribution_action=None)
    train_loader, val_loader, OoD_val_loader, test_loader = data.get_dataloaders(train_batch=train_batch_size, test_batch=test_batch_size, job=job)
    input_n=data.node_n
print(">>> data loaded !")
# ===============================================================
# Instantiate model, and methods used fro training and valdation
# ===============================================================

model = nnmodel.VAE(input_n=input_n, hidden_layers=opt.hidden_layers,  n_z=opt.n_z, variational=opt.variational, output_variance=opt.output_variance, device=device, batch_norm=opt.batch_norm, p_dropout=opt.p_drop)
model._initialise(start_epoch=opt.start_epoch, folder_name=folder_name, lr=opt.lr, beta=opt.beta, l2_reg=weight_decay, train_batch_size=train_batch_size)

for epoch in range(opt.start_epoch, opt.n_epochs+1):
    print("Epoch:{}/{}".format(epoch, opt.n_epochs))

    if opt.use_MNIST:
        model.train_epoch_mnist(epoch, train_loader, opt.use_bernoulli_loss)
        model.eval_full_batch_mnist(train_loader, epoch, 'train', opt.use_bernoulli_loss)
        model.eval_full_batch_mnist(val_loader, epoch, 'val', opt.use_bernoulli_loss)
    else:
        model.train_epoch(epoch, train_loader)
        model.eval_full_batch(train_loader, epoch, 'train')
        model.eval_full_batch(val_loader, epoch, 'val')

    model.save_checkpoint_and_csv(epoch)







