import experiments.utils as utils
import torch.optim
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.stats import norm

# ===============================================================
#                     Degradations
# ===============================================================
def degradation_experiments_nsamples(model, acts, val_loader, alpha=0, num_occlusions=0):
    first_loop=True
    for num_samples in [2**i for i in range(8)]:
        val_mse_accum = []
        for i in range(10):
            for act in acts:
                for i, (all_seq) in enumerate(val_loader[act]):
                    inputs = all_seq
                    inputs_degraded = utils.simulate_occlusions(inputs, num_occlusions=num_occlusions, folder_name="")
                    inputs_degraded_noise = utils.add_noise(inputs_degraded, alpha=alpha)
                    inputs_degraded_noise = torch.from_numpy(inputs_degraded_noise).to(model.device).float()

                    _ = model(inputs_degraded_noise.float(), num_samples=num_samples)
                    _ = model.loss_VLB(inputs_degraded_noise)

                    model.accum_update('val_mse_loss', model.recon_loss)

            val_mse_accum.append(model.accum_loss['val_mse_loss'].avg)

        print("Validation, expectation taken over {} samples, reconstruction (to gt) MSE:{} +- {}".format(num_samples, np.mean(val_mse_accum), np.std(val_mse_accum)))

        head = ['num_samples', 'MSE', 'STD']
        ret_log = [num_samples, np.mean(val_mse_accum), np.std(val_mse_accum)]
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
        if first_loop:
            df.to_csv(model.folder_name+'/'+'nsamples.csv', header=head, index=False)
            first_loop=False
        else:
            with open(model.folder_name+'/'+'nsamples.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)

def degradation_experiments_occlude(model, acts, val_loader, alpha=0, num_samples=1):
    first_loop=True
    for num_occlusions in range(0, 97, 3):
        val_mse_accum = []
        for i in range(10):
            for act in acts:
                for i, (all_seq) in enumerate(val_loader[act]):
                    inputs = all_seq
                    inputs_degraded = utils.simulate_occlusions(inputs, num_occlusions=num_occlusions, folder_name="")
                    inputs_degraded_noise = utils.add_noise(inputs_degraded, alpha=alpha)
                    inputs_degraded_noise = torch.from_numpy(inputs_degraded_noise).to(model.device).float()

                    _ = model(inputs_degraded_noise.float(), num_samples=num_samples)
                    _ = model.loss_VLB(inputs_degraded_noise)

                    model.accum_update('val_mse_loss', model.recon_loss)

            val_mse_accum.append(model.accum_loss['val_mse_loss'].avg)

        print("Validation, degraded by {} occlutions, reconstruction (to gt) MSE:{} +- {}".format(num_occlusions, np.mean(val_mse_accum), np.std(val_mse_accum)))

        head = ['num_occlusions', 'MSE', 'STD']
        ret_log = [num_occlusions, np.mean(val_mse_accum), np.std(val_mse_accum)]
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
        if first_loop:
            df.to_csv(model.folder_name+'/'+'occlusions_nsamp'+str(num_samples)+'.csv', header=head, index=False)
            first_loop=False
        else:
            with open(model.folder_name+'/'+'occlusions_nsamp'+str(num_samples)+'.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)

def degradation_experiments_noise(model, acts, val_loader, num_occlusions=0, num_samples=1):
    noise_scale_range = [0.0] + [1.5**i for i in range(-10,5)]
    first_loop=True
    for alpha in noise_scale_range:
        val_mse_accum = []
        for i in range(10):
            for act in acts:
                for i, (all_seq) in enumerate(val_loader[act]):
                    inputs = all_seq
                    inputs_degraded = utils.simulate_occlusions(inputs, num_occlusions=num_occlusions, folder_name="")
                    inputs_degraded_noise = utils.add_noise(inputs_degraded, alpha=alpha)
                    inputs_degraded_noise = torch.from_numpy(inputs_degraded_noise).to(model.device).float()

                    _ = model(inputs_degraded_noise.float(), num_samples=num_samples)
                    _ = model.loss_VLB(inputs_degraded_noise)

                    model.accum_update('val_mse_loss', model.recon_loss)

            val_mse_accum.append(model.accum_loss['val_mse_loss'].avg)

        print("Validation, degraded by noise of scale {}, reconstruction (to gt) MSE:{} +- {}".format(alpha, np.mean(val_mse_accum), np.std(val_mse_accum)))

        head = ['alpha_noise', 'MSE', 'STD']
        ret_log = [alpha, np.mean(val_mse_accum), np.std(val_mse_accum)]
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
        if first_loop:
            df.to_csv(model.folder_name+'/'+'noise_nsamp'+str(num_samples)+'.csv', header=head, index=False)
            first_loop=False
        else:
            with open(model.folder_name+'/'+'noise_nsamp'+str(num_samples)+'.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)

# ===============================================================
#                     Embeddings
# ===============================================================
def de_noise(model, acts, data_loader, num_occlusions=0, alpha=0):
    recons = dict()
    df_recons = pd.DataFrame()
    for act in acts:
        recons[act] = []
        for i, (all_seq) in enumerate(data_loader[act]):
            inputs = all_seq
            inputs_degraded = utils.simulate_occlusions(inputs, num_occlusions=num_occlusions, folder_name="")
            inputs_degraded_noise = utils.add_noise(inputs_degraded, alpha=alpha)
            inputs_degraded_noise = torch.from_numpy(inputs_degraded_noise).to(model.device).float()

            recons = model(inputs_degraded_noise)

            if recons[act] == []:
                recons[act] = mu.detach().cpu().numpy()
            else:
                recons[act] = np.vstack((recons[act], mu.detach().cpu().numpy()))
            if df_recons.empty:
                df = pd.DataFrame(mu.detach().cpu().numpy())
                df['action'] = act
                df_recons = df
            else:
                df = pd.DataFrame(mu.detach().cpu().numpy())
                df['action'] = act
                df_recons = df_recons.append(df)

    return recons, df_recons

def embed(model, acts, data_loader, num_occlusions=0, alpha=0):
    embeddings = dict()
    df_embeddings = pd.DataFrame()
    for act in acts:
        embeddings[act] = []
        for i, (all_seq) in enumerate(data_loader[act]):
            inputs = all_seq
            inputs_degraded = utils.simulate_occlusions(inputs, num_occlusions=num_occlusions, folder_name="")
            inputs_degraded_noise = utils.add_noise(inputs_degraded, alpha=alpha)
            inputs_degraded_noise = torch.from_numpy(inputs_degraded_noise).to(model.device).float()

            mu, _ = model.encoder(inputs_degraded_noise)

            if embeddings[act] == []:
                embeddings[act] = mu.detach().cpu().numpy()
            else:
                embeddings[act] = np.vstack((embeddings[act], mu.detach().cpu().numpy()))
            if df_embeddings.empty:
                df = pd.DataFrame(mu.detach().cpu().numpy())
                df['action'] = act
                df_embeddings = df
            else:
                df = pd.DataFrame(mu.detach().cpu().numpy())
                df['action'] = act
                df_embeddings = df_embeddings.append(df)

    return embeddings, df_embeddings

def inputs(acts, data_loader, num_occlusions=0, alpha=0):
    """

    :param data_loader: torch dataloader object
    :param num_occlusions: number of occlusions to make per datapoint (int)
    :param alpha: scale of gaussian noise to add (float)
    :return: pandas dataframe of the inputs, labelled with added noise and occlusions (num_datapoints, num_features + label)
    """
    df_inputs = pd.DataFrame()
    for act in acts:
        for i, (all_seq) in enumerate(data_loader[act]):
            inputs = all_seq

            inputs_degraded = utils.simulate_occlusions(inputs, num_occlusions=num_occlusions, folder_name="")
            inputs_degraded_noise = utils.add_noise(inputs_degraded, alpha=alpha)
            inputs_degraded_noise = torch.from_numpy(inputs_degraded_noise)

            if df_inputs.empty:
                df = pd.DataFrame(inputs_degraded_noise.detach().cpu().numpy())
                df['action'] = act
                df_inputs = df
            else:
                df = pd.DataFrame(inputs_degraded_noise.detach().cpu().numpy())
                df['action'] = act
                df_inputs = df_inputs.append(df)

    return df_inputs


def plot_embedding_rainbow(folder_name, embeddings, acts, cdf_plot=False):
    if cdf_plot:
        embeddings = norm.cdf(embeddings)

    fig = plt.figure()
    fig = plt.figure(figsize=(40, 40))

    colors = cm.rainbow(np.linspace(0, 1, len(acts)))
    i=0
    for act in acts:
        plt.scatter(embeddings[act][:, 0], embeddings[act][:, 1], s=2, marker='o', alpha=0.1, color=colors[i])
        embedding_act_avg = np.mean(embeddings[act], axis=0)
        plt.scatter(embedding_act_avg[0], embedding_act_avg[1], s=2000, marker='x', alpha=1.0, color=colors[i], label=act)
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
    leg = plt.legend(prop={'size': 30})
    for lh in leg.legendHandles:
        lh.set_alpha(1.0)
    plt.savefig(folder_name + '/embedding_rainbow')

def interpolate_acts(model, embeddings, act1, act2):
    embedding_act1_avg = np.mean(embeddings[act1], axis=0)
    embedding_act2_avg = np.mean(embeddings[act2], axis=0)

    vec1to2 = embedding_act2_avg - embedding_act1_avg

    inter_points = embedding_act1_avg
    points = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for step_length in points:
        inter_points = np.vstack((inter_points, embedding_act1_avg + step_length*vec1to2))

    inputs = torch.from_numpy(inter_points).to(model.device)
    mu = model.generate(inputs.float())

    file_path = model.folder_name + '/' + 'interposations_xz'
    utils.plot_poses(mu.detach().cpu().numpy(), mu.detach().cpu().numpy(), max_num_images=25, azim=0, evl=90, save_as=file_path, one_dim_grid=True)
    file_path = model.folder_name + '/' + 'interposations_yz'
    utils.plot_poses(mu.detach().cpu().numpy(), mu.detach().cpu().numpy(), max_num_images=25, azim=0, evl=-0, save_as=file_path, one_dim_grid=True)
    file_path = model.folder_name + '/' + 'interposations_xy'
    utils.plot_poses(mu.detach().cpu().numpy(), mu.detach().cpu().numpy(), max_num_images=25, azim=90, evl=90, save_as=file_path, one_dim_grid=True)


# ===============================================================
#                     ICDF
# ===============================================================
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