from __future__ import print_function, absolute_import, division

import torch.optim

from data import DATA

import models.VAE as nnmodel

import experiments.experiments as experiments

from opt import Options
opt = Options().parse()

folder_name=opt.name
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = "cuda"
    job = 10 #Can change this to multi-process dataloading
else:
    device = "cpu"
    job = 0

##################################################################
# Load Data
##################################################################
if not opt.icdf:
    train_batch_size=opt.train_batch_size
    test_batch_size=opt.test_batch_size
    data = DATA("h3.6m_3d", "h3.6m/dataset/")
    out_of_distribution = data.get_poses(input_n=1, output_n=1, sample_rate=2, dct_n=2, out_of_distribution_action=None, val_categorise=True)
    train_loader, val_loader, OoD_val_loader, test_loader = data.get_dataloaders(train_batch=train_batch_size, test_batch=test_batch_size, job=job, val_categorise=True)

##################################################################
# Instantiate model, and methods used fro training and valdation
##################################################################

model = nnmodel.VAE(input_n=96, hidden_layers=opt.hidden_layers, n_z=opt.n_z, variational=opt.variational, output_variance=opt.output_variance, device=device, batch_norm=opt.batch_norm, p_dropout=opt.p_drop)
model._initialise(start_epoch=opt.start_epoch, folder_name=folder_name)
model.eval()

if opt.degradation_experiment:
    #experiments.degradation_experiments_occlude(model, data.acts_train, val_loader, alpha=0, num_samples=10)
    experiments.degradation_experiments_noise(model, data.acts_train, val_loader, num_occlusions=0, num_samples=10)
    #experiments.degradation_experiments_nsamples(model, data.acts_train, val_loader, alpha=0, num_occlusions=0)

if opt.embedding_experiment:
    embeddings, df_embeddings = experiments.embed(model, data.acts_train, train_loader)

    #experiments.plot_embedding_rainbow(model.folder_name, embeddings, data.acts_train, cdf_plot=False)
    #experiments.plot_embedding_rainbow(model.folder_name, embeddings, ['walking', 'walkingtogether'], cdf_plot=False)
    #experiments.interpolate_acts(model, embeddings, 'walking', 'walkingtogether')

if opt.de_noise:
    alphas = [0.0, 0.1, 0.3, 1.0]
    for alpha in alphas:
        recons, df_recons = experiments.embed(model, data.acts_train, train_loader, num_occlusions=0, alpha=alpha)
        df_recons.to_csv(model.folder_name + '/recons/' + 'noise_' + str(alpha) + '/' + 'train.csv', header=False, index=False)
        recons, df_recons = experiments.embed(model, data.acts_train, val_loader, num_occlusions=0, alpha=alpha)
        df_recons.to_csv(model.folder_name + '/recons/' + 'noise_' + str(alpha) + '/' + 'val.csv', header=False, index=False)
        recons, df_recons = experiments.embed(model, data.acts_train, test_loader, num_occlusions=0, alpha=alpha)
        df_recons.to_csv(model.folder_name + '/recons/' + 'noise_' + str(alpha) + '/' + 'test.csv', header=False, index=False)

if opt.noise_to_embeddings:
    alphas = [0.0, 0.1, 0.3, 1.0]
    for alpha in alphas:
        embeddings, df_embeddings = experiments.embed(model, data.acts_train, train_loader, num_occlusions=0, alpha=alpha)
        df_embeddings.to_csv(model.folder_name + '/embeddings/' + 'noise_' + str(alpha) + '/' + 'train.csv', header=False, index=False)
        embeddings, df_embeddings = experiments.embed(model, data.acts_train, val_loader, num_occlusions=0, alpha=alpha)
        df_embeddings.to_csv(model.folder_name + '/embeddings/' + 'noise_' + str(alpha) + '/' + 'val.csv', header=False, index=False)
        embeddings, df_embeddings = experiments.embed(model, data.acts_train, test_loader, num_occlusions=0, alpha=alpha)
        df_embeddings.to_csv(model.folder_name + '/embeddings/' + 'noise_' + str(alpha) + '/' + 'test.csv', header=False, index=False)

if opt.noise_to_inputs:
    alphas = [0.0, 0.1, 0.3, 1.0]
    for alpha in alphas:
        df_inputs = experiments.inputs(data.acts_train, train_loader, num_occlusions=0, alpha=alpha)
        df_inputs.to_csv('inputs' + '/' + 'noise_' + str(alpha) + '/' + 'train.csv', header=False, index=False)
        df_inputs = experiments.inputs(data.acts_train, val_loader, num_occlusions=0, alpha=alpha)
        df_inputs.to_csv('inputs' + '/' + 'noise_' + str(alpha) + '/' + 'val.csv', header=False, index=False)
        df_inputs = experiments.inputs(data.acts_train, test_loader, num_occlusions=0, alpha=alpha)
        df_inputs.to_csv('inputs' + '/' + 'noise_' + str(alpha) + '/' + 'test.csv', header=False, index=False)

if opt.icdf:
    experiments.gnerate_icdf(model, opt.grid_size)



