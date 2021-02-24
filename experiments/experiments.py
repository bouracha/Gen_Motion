import models.utils as utils
import torch.optim
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.stats import norm

def degradation_experiments(model, acts, val_loader):
    noise_scale_range = [0.0] + [1.5**i for i in range(-10,5)]
    num_occlusions=0
    alpha=0
    first_loop=True
    #for num_occlusions in range(0, 97, 3):
    for alpha in noise_scale_range:
        val_mse_accum = []
        for i in range(10):
            for act in acts:
                for i, (all_seq) in enumerate(val_loader[act]):
                    cur_batch_size = len(all_seq)

                    inputs = all_seq
                    inputs_degraded = utils.simulate_occlusions(inputs, num_occlusions=num_occlusions, folder_name="")
                    inputs_degraded_noise = utils.add_noise(inputs_degraded, alpha=alpha)
                    inputs_degraded_noise = torch.from_numpy(inputs_degraded_noise).to(model.device).float()

                    mu = model(inputs_degraded_noise.float())
                    loss = model.loss_VLB(inputs_degraded_noise)

                    model.accum_update('val_loss', loss)
                    model.accum_update('val_mse_loss', model.recon_loss)

            val_mse_accum.append(model.accum_loss['val_mse_loss'].avg)

        #print("Validation, degraded by {} occlutions, reconstruction (to gt) MSE:{} +- {}".format(num_occlusions, np.mean(val_mse_accum), np.std(val_mse_accum)))
        print("Validation, degraded by noise of scale {}, reconstruction (to gt) MSE:{} +- {}".format(alpha, np.mean(val_mse_accum), np.std(val_mse_accum)))

        head = ['num_occlusions', 'MSE', 'STD']
        #ret_log = [num_occlusions, np.mean(val_mse_accum), np.std(val_mse_accum)]
        ret_log = [alpha, np.mean(val_mse_accum), np.std(val_mse_accum)]
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
        file_name = "added_noise.csv"
        if first_loop:
            df.to_csv(model.folder_name+'/'+str(file_name), header=head, index=False)
            first_loop=False
        else:
            with open(model.folder_name+'/'+str(file_name), 'a') as f:
                df.to_csv(f, header=False, index=False)

def embed(model, acts, data_loader):
    embeddings = dict()
    num_outliers=0
    for act in acts:
        embeddings[act] = np.array([0,0]).reshape((1,2))
        for i, (all_seq) in enumerate(data_loader[act]):

            inputs = all_seq.to(model.device).float()

            mu, z, KL = model.encoder(inputs)

            #embeddings[act] = np.vstack((embeddings[act], norm.cdf(mu.detach().cpu().numpy())))
            embeddings[act] = np.vstack((embeddings[act], mu.detach().cpu().numpy()))

            #for i in range(int(mu.shape[0])):
            #    if (mu[i, 1] > 8):
            #        num_outliers+=1
            #        file_path = model.folder_name + '/pose_outliers/' + str(act) + '_' + 'poses_xz_'+str(num_outliers)
            #        model.plot_poses(inputs[i], inputs[i], num_images=1, azim=0, evl=90, save_as=file_path)

    print(embeddings['walking'].shape)

    alpha = 0.1
    scale = 3

    fig = plt.figure()
    fig = plt.figure(figsize=(20, 20))

    colors = cm.rainbow(np.linspace(0, 1, 13))
    i=0
    for act in acts:
        plt.scatter(embeddings[act][:, 0], embeddings[act][:, 1], s=scale, marker='o', alpha=alpha, color=colors[i], label=act)
        i+=1
    #plt.scatter(embeddings['walking'][:, 0], embeddings['walking'][:, 1], s=scale, marker='o', alpha=alpha, color='b', label='walking')
    #plt.scatter(embeddings['walkingdog'][:, 0], embeddings['walkingdog'][:, 1], s=scale, marker='o', alpha=alpha, color='b', label='walkingdog')
    #plt.scatter(embeddings['walkingtogether'][:, 0], embeddings['walkingtogether'][:, 1], s=scale, marker='o', alpha=alpha, color='b', label='walkingtogether')
    #plt.scatter(embeddings['eating'][:, 0], embeddings['eating'][:, 1], s=scale, marker='o', alpha=alpha, color='g', label='eating')
    #plt.scatter(embeddings['smoking'][:, 0], embeddings['smoking'][:, 1], s=scale, marker='o', alpha=alpha, color='y', label='smoking')
    #plt.scatter(embeddings['discussion'][:, 0], embeddings['discussion'][:, 1], s=scale, marker='o', alpha=alpha, color='c', label='discussion')
    #plt.scatter(embeddings['directions'][:, 0], embeddings['directions'][:, 1], s=scale, marker='o', alpha=alpha, color='k', label='directions')
    #plt.scatter(embeddings['phoning'][:, 0], embeddings['phoning'][:, 1], s=scale, marker='o', alpha=alpha, color='m', label='phoning')
    #plt.scatter(embeddings['sitting'][:, 0], embeddings['sitting'][:, 1], s=scale, marker='o', alpha=alpha, color='r', label='sitting')
    #plt.scatter(embeddings['sittingdown'][:, 0], embeddings['sittingdown'][:, 1], s=scale, marker='o', alpha=alpha, color='m', label='sittingdown')

    plt.legend()

    plt.savefig(model.folder_name + '/Embedding')


def gnerate_icdf(model, num_grid_points=20):
    z = np.random.randn(num_grid_points ** 2, 2)
    linspace = np.linspace(0.01, 0.99, num=num_grid_points)
    count = 0
    for i in linspace:
        for j in linspace:
            z[count, 0] = j
            z[count, 1] = i
            count += 1

    z = norm.ppf(z)
    inputs = torch.from_numpy(z).to(model.device)
    mu = model.generate(inputs.float())

    file_path = model.folder_name + '/' + 'poses_xz'
    utils.plot_poses(mu.detach().cpu().numpy(), mu.detach().cpu().numpy(), max_num_images=num_grid_points ** 2, azim=0, evl=90, save_as=file_path)
    file_path = model.folder_name + '/' + 'poses_yz'
    utils.plot_poses(mu.detach().cpu().numpy(), mu.detach().cpu().numpy(), max_num_images=num_grid_points ** 2, azim=0, evl=-0, save_as=file_path)
    file_path = model.folder_name + '/' + 'poses_xy'
    utils.plot_poses(mu.detach().cpu().numpy(), mu.detach().cpu().numpy(), max_num_images=num_grid_points ** 2, azim=90, evl=90, save_as=file_path)