import utils.data_utils as data_utils
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from utils.h36motion3d import H36motion3D_pose

import utils.viz_3d as viz_3d

import utils.utils as utils

acts = data_utils.define_actions("all")

train_dataset = H36motion3D_pose(path_to_data="h3.6m/dataset/", actions=acts, input_n=1, output_n=1, split=0, sample_rate=2, dct_used=2)
val_dataset = H36motion3D_pose(path_to_data="h3.6m/dataset/", actions=acts, input_n=1, output_n=1, split=2, sample_rate=2, dct_used=2)

X = train_dataset.all_seqs
X_val = val_dataset.all_seqs

print(X.shape)


multi_dim_experiment = False
if multi_dim_experiment:
    for n_z in [2, 3, 5, 10, 20, 48]:
        pca = PCA(n_components=n_z)
        z = pca.fit_transform(X)

        z_val = pca.transform(X_val)

        X_recon = pca.inverse_transform(z)
        X_recon_val = pca.inverse_transform(z_val)

        train_mse = np.mean(np.sum((X_recon - X)**2, axis=-1))
        val_mse = np.mean(np.sum((X_recon_val - X_val)**2, axis=-1))

        print("Train reconstruction MSE:{}".format(train_mse))
        print("Validation reconstruction MSE:{}".format(val_mse))

    avg_pose = np.mean(X, axis=0)
    avg_pose_repeated = np.repeat(avg_pose.reshape(1, -1), X_val.shape[0], axis=0)

    val_mse_avg = np.mean(np.sum((avg_pose_repeated - X_val) ** 2, axis=-1))
    print("Validation average to gt MSE:{}".format(val_mse_avg))

first_loop=True
for num_occlusions in range(0, 97, 3):
    val_mse_accum = []
    for i in range(100):
        pca = PCA(n_components=2)
        z = pca.fit_transform(X)

        X_val_degraded = np.copy(X_val)
        X_val_degraded = utils.simulate_occlusions(X_val_degraded, num_occlusions=num_occlusions, folder_name="")

        z_val_degraded = pca.transform(X_val_degraded)

        X_recon_val_degraded = pca.inverse_transform(z_val_degraded)

        val_mse = np.mean(np.sum((X_recon_val_degraded - X_val) ** 2, axis=-1))

        val_mse_accum.append(val_mse)

    print("Validation, degraded by {} occlutions, reconstruction (to gt) MSE:{} +- {}".format(num_occlusions, np.mean(val_mse_accum), np.std(val_mse_accum)))

    head = ['num_occlusions', 'MSE', 'STD']
    ret_log = [num_occlusions, np.mean(val_mse_accum), np.std(val_mse_accum)]
    df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
    if first_loop:
        df.to_csv('pca_occlusions.csv', header=head, index=False)
        first_loop=False
    else:
        with open('pca_occlusions.csv', 'a') as f:
            df.to_csv(f, header=False, index=False)





plot = False
if plot:
    val_idx = np.random.randint(val_dataset.all_seqs.shape[0], size=25)
    file_path = 'pca_poses_yz'
    utils.plot_poses(X[val_idx], X_recon[val_idx], num_images=25, azim=0, evl=-0, save_as=file_path)
    file_path = 'pca_poses_xz'
    utils.plot_poses(X[val_idx], X_recon[val_idx], num_images=25, azim=0, evl=90, save_as=file_path)
    file_path = 'pca_poses_xy'
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

        z_act = pca.transform(X_act)

        plt.scatter(z_act[:, 0], z_act[:, 1], s=scale, marker='o', alpha=alpha, color=colors[i], label=act)
        i+=1

    plt.legend()
    plt.savefig('PCA_embedding')