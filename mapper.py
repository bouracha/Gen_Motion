import utils.data_utils as data_utils
import umap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from utils.h36motion3d import H36motion3D_pose

import utils.viz_3d as viz_3d

import models.utils as utils

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--multi_dim_experiment', dest='multi_dim_experiment', action='store_true', help='toggle do multi-dim experiments')
parser.add_argument('--degradation_experiment', dest='degradation_experiment', action='store_true', help='toggle do degradation experiments')
parser.add_argument('--n_z', type=int, default=2, help='Number of latent variables')
parser.add_argument('--algorithm', type=str, default="PCA", help='choose PCA or UMAP')
parser.set_defaults(multi_dim_experiment=False)
parser.set_defaults(degradation_experiment=False)

opt = parser.parse_args()


acts = data_utils.define_actions("all")

train_dataset = H36motion3D_pose(path_to_data="h3.6m/dataset/", actions=acts, input_n=1, output_n=1, split=0, sample_rate=2, dct_used=2)
val_dataset = H36motion3D_pose(path_to_data="h3.6m/dataset/", actions=acts, input_n=1, output_n=1, split=2, sample_rate=2, dct_used=2)

X = train_dataset.all_seqs
X_val = val_dataset.all_seqs

if opt.algorithm.upper() == "PCA":
    algorithm = "pca"
elif opt.algorithm.upper() == "UMAP":
    algorithm = "umap"
else:
    print("{} is not a valid algorithm option".format(opt.algorithm.upper()))

print(X.shape)

multi_dim_experiment = opt.multi_dim_experiment
if multi_dim_experiment:
    for n_z in [2, 3, 5, 10, 20, 48]:
        if algorithm.upper() == "PCA":
            mapper = PCA(n_components=n_z)
        elif algorithm.upper() == "UMAP":
            mapper = umap.UMAP(random_state=42, n_components=n_z)
        else:
            print("{} is not a valid algorithm option".format(opt.algorithm.upper()))
        z = mapper.fit_transform(X)

        z_val = mapper.transform(X_val)

        X_recon = mapper.inverse_transform(z)
        X_recon_val = mapper.inverse_transform(z_val)

        train_mse = np.mean(np.sum((X_recon - X)**2, axis=-1))
        val_mse = np.mean(np.sum((X_recon_val - X_val)**2, axis=-1))

        print("Train reconstruction MSE:{}".format(train_mse))
        print("Validation reconstruction MSE:{}".format(val_mse))

    avg_pose = np.mean(X, axis=0)
    avg_pose_repeated = np.repeat(avg_pose.reshape(1, -1), X_val.shape[0], axis=0)

    val_mse_avg = np.mean(np.sum((avg_pose_repeated - X_val) ** 2, axis=-1))
    print("Validation average to gt MSE:{}".format(val_mse_avg))

degradation_experiment = opt.degradation_experiment
if degradation_experiment:
    if algorithm.upper() == "PCA":
        mapper = PCA(n_components=opt.n_z)
    elif algorithm.upper() == "UMAP":
        mapper = umap.UMAP(random_state=42, n_components=opt.n_z)
    else:
        print("{} is not a valid algorithm option".format(opt.algorithm.upper()))
    z = mapper.fit(X)
    # noise_scale_range_log_1 = [2**i for i in range(-5,-1)]
    # noise_scale_range_linear = [0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0]
    # noise_scale_range_log_2 = [2**i for i in range(1,5)]
    # noise_scale_range = noise_scale_range_log_1 + noise_scale_range_linear + noise_scale_range_log_2
    # noise_scale_range = [i for i in range(0.0, 2.0, 0.)]
    noise_scale_range = [0.0] + [1.5 ** i for i in range(-10, 5)]
    num_occlusions = 0
    alpha = 0
    first_loop=True
    #for num_occlusions in range(0, 97, 3):
    for alpha in noise_scale_range:
        val_mse_accum = []
        for i in range(10):

            X_val_degraded = utils.simulate_occlusions(X_val, num_occlusions=num_occlusions, folder_name="")
            X_val_degraded_noise = utils.add_noise(X_val_degraded, alpha=alpha)

            z_val_degraded = mapper.transform(X_val_degraded_noise)

            X_recon_val_degraded = mapper.inverse_transform(z_val_degraded)

            val_mse = np.mean(np.sum((X_recon_val_degraded - X_val) ** 2, axis=-1))

            val_mse_accum.append(val_mse)

        # print("Validation, degraded by {} occlutions, reconstruction (to gt) MSE:{} +- {}".format(num_occlusions, np.mean(val_mse_accum), np.std(val_mse_accum)))
        print("Validation, degraded by noise of scale {}, reconstruction (to gt) MSE:{} +- {}".format(alpha, np.mean(val_mse_accum), np.std(val_mse_accum)))

        head = ['num_occlusions', 'MSE', 'STD']
        # ret_log = [num_occlusions, np.mean(val_mse_accum), np.std(val_mse_accum)]
        ret_log = [alpha, np.mean(val_mse_accum), np.std(val_mse_accum)]
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
        file_name = "added_noise_"+str(opt.n_z)
        if first_loop:
            df.to_csv(str(algorithm)+'/'+str(file_name)+'.csv', header=head, index=False)
            first_loop=False
        else:
            with open(str(algorithm)+'/'+str(file_name)+'.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)



plot = False
if plot:
    val_idx = np.random.randint(val_dataset.all_seqs.shape[0], size=25)
    file_path = 'umap_poses_yz'
    utils.plot_poses(X[val_idx], X_recon[val_idx], num_images=25, azim=0, evl=-0, save_as=file_path)
    file_path = 'umap_poses_xz'
    utils.plot_poses(X[val_idx], X_recon[val_idx], num_images=25, azim=0, evl=90, save_as=file_path)
    file_path = 'umap_poses_xy'
    utils.plot_poses(X[val_idx], X_recon[val_idx], num_images=25, azim=90, evl=90, save_as=file_path)

    alpha = 1.0
    scale = 3
    fig = plt.figure()
    fig = plt.figure(figsize=(20, 20))
    colors = cm.rainbow(np.linspace(0, 1, 13))
    i=0
    for act in acts:
        X_act, _, _ = data_utils.load_data_3d("h3.6m/dataset/", [1, 6, 7, 8, 9], [act], sample_rate=2, seq_len=1)
        #X_act_val, _, _ = data_utils.load_data_3d("h3.6m/dataset/", [11], act, sample_rate=2, seq_len=1)

        z_act = mapper.transform(X_act)

        plt.scatter(z_act[:, 0], z_act[:, 1], s=scale, marker='o', alpha=alpha, color=colors[i], label=act)
        i+=1

    plt.legend()
    plt.savefig('UMAP_embedding')