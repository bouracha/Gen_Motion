from __future__ import print_function, absolute_import, division

import torch.optim

from data import DATA

import models.VAE as nnmodel

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--variational', dest='variational', action='store_true', help='toggle VAE or AE')
parser.add_argument('--batch_norm', dest='batch_norm', action='store_true', help='toggle use batch_norm or not')
parser.add_argument('--weight_decay', dest='weight_decay', action='store_true', help='toggle use weight decay or not')
parser.add_argument('--output_variance', dest='output_variance', action='store_true', help='toggle model output variance or use as constant')
parser.add_argument('--use_MNIST', dest='use_MNIST', action='store_true', help='toggle to use MNIST data instead')
parser.add_argument('--use_bernoulli_loss', dest='use_bernoulli_loss', action='store_true', help='toggle to bernoulli of gauss loss')
parser.add_argument('--beta', type=float, default=1.0, help='Downweighting of the KL divergence')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--n_z', type=int, default=2, help='Number of latent variables')
parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs to train for')
parser.add_argument('--start_epoch', type=int, default=1, help='If not 1, load checkpoint at this epoch')
parser.add_argument('--train_batch_size', type=int, default=100, help='Number of epochs to train for')
parser.add_argument('--test_batch_size', type=int, default=100, help='If not 1, load checkpoint at this epoch')
parser.add_argument('--name', type=str, default="", help='Name of master folder containing model')
#parser.add_argument('--n_epochs_per_save', type=int, default=10, help='Number of epochs before saving the checkpoints')
parser.add_argument('--encoder_hidden_layers', nargs='+', type=int, default=[500, 200, 100, 50], help='input the out of distribution action')
parser.set_defaults(variational=False)
parser.set_defaults(batch_norm=False)
parser.set_defaults(weight_decay=False)
parser.set_defaults(output_variance=False)
parser.set_defaults(use_MNIST=False)
parser.set_defaults(use_bernoulli_loss=False)

opt = parser.parse_args()

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

#####################################################
# Load data
#####################################################
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
else:
    ### Human Motion Data
    data = DATA("h3.6m_3d", "h3.6m/dataset/")
    out_of_distribution = data.get_poses(input_n=1, output_n=1, sample_rate=2, dct_n=2, out_of_distribution_action=None)
    train_loader, val_loader, OoD_val_loader, test_loader = data.get_dataloaders(train_batch=train_batch_size, test_batch=test_batch_size, job=job)
    print(">>> train data {}".format(data.train_dataset.__len__()))
    print(">>> validation data {}".format(data.val_dataset.__len__()))

print(">>> data loaded !")
##################################################################
# Instantiate model, and methods used fro training and valdation
##################################################################
input_n = data.node_n
n_z = opt.n_z
n_epochs = opt.n_epochs
lr=opt.lr
encoder_hidden_layers = opt.encoder_hidden_layers
start_epoch = opt.start_epoch
batch_norm = opt.batch_norm

encoder_layers = []
decoder_layers = []
encoder_layers.append(input_n)
decoder_layers.append(n_z)
n_hidden = len(encoder_hidden_layers)
for i in range(n_hidden):
    encoder_layers.append(encoder_hidden_layers[i])
    decoder_layers.append(encoder_hidden_layers[n_hidden-1-i])
encoder_layers.append(n_z)
decoder_layers.append(input_n)

print(">>> creating model")
model = nnmodel.VAE(encoder_layers=encoder_layers,  decoder_layers=decoder_layers, lr=lr, train_batch_size=train_batch_size, variational=opt.variational, output_variance=opt.output_variance, device=device, batch_norm=batch_norm, weight_decay=weight_decay, p_dropout=0.0, beta=opt.beta, start_epoch=start_epoch, folder_name=folder_name)
clipping_value = 1
torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
if is_cuda:
    model.cuda()
print(model)

print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
if start_epoch != 1:
    model.book_keeping(model.encoder_layers, model.decoder_layers, start_epoch=start_epoch, lr=lr, train_batch_size=train_batch_size, batch_norm=batch_norm, weight_decay=weight_decay, p_dropout=0.0)
    ckpt_path = model.folder_name + '/checkpoints/' + 'ckpt_' + str(start_epoch-1) + '_weights.path.tar'
    ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])

for epoch in range(start_epoch, n_epochs+1):
    print("Epoch: ", epoch)

    if opt.use_MNIST:
        model.train_epoch_mnist(epoch, train_loader, optimizer, opt.use_bernoulli_loss)
        model.eval_full_batch_mnist(train_loader, epoch, 'train', opt.use_bernoulli_loss)
        model.eval_full_batch_mnist(val_loader, epoch, 'val', opt.use_bernoulli_loss)
    else:
        model.train_epoch(epoch, train_loader, optimizer)
        model.eval_full_batch(train_loader, epoch, 'train')
        model.eval_full_batch(val_loader, epoch, 'val')

    model.save_checkpoint_and_csv(epoch, lr, optimizer)







