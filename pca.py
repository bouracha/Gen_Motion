import utils.data_utils as data_utils
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from utils.h36motion3d import H36motion3D_pose

import utils.viz_3d as viz_3d

acts = data_utils.define_actions("all")

train_dataset = H36motion3D_pose(path_to_data="h3.6m/dataset/", actions=acts, input_n=1, output_n=1, split=0, sample_rate=2, dct_used=2)
val_dataset = H36motion3D_pose(path_to_data="h3.6m/dataset/", actions=acts, input_n=1, output_n=1, split=2, sample_rate=2, dct_used=2)

X = train_dataset.all_seqs
X_val = val_dataset.all_seqs

print(X.shape)

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


def plot_poses(xyz_gt, xyz_pred, num_images=25, azim=0, evl=0, save_as=None):
    '''
    Function for visualizing poses: saves grid of poses.
    Assumes poses are normalised between 0 and 1
    :param xyz_gt: set of ground truth 3D joint positions (batch_size, 96)
    :param xyz_pred: set of predicted 3D joint positions (batch_size, 96)
    :param num_images: number of poses to plotted from given set (int)
    :param azim: azimuthal angle for viewing (int)
    :param evl: angle of elevation for viewing (int)
    :param save_as: path and name to save (str)
    '''
    xyz_gt = xyz_gt.reshape(-1, 96)
    xyz_pred = xyz_pred.reshape(-1, 96)

    xyz_gt = xyz_gt[:num_images].reshape(num_images, 32, 3)
    xyz_pred = xyz_pred[:num_images].reshape(num_images, 32, 3)

    fig = plt.figure()
    if num_images > 4:
        fig = plt.figure(figsize=(20, 20))
    if num_images > 40:
        fig = plt.figure(figsize=(50, 50))

    grid_dim_size = np.ceil(np.sqrt(num_images))
    for i in range(num_images):
        ax = fig.add_subplot(grid_dim_size, grid_dim_size, i + 1, projection='3d')
        ax.set_xlim3d([0, 1])
        ax.set_ylim3d([0, 1])
        ax.set_zlim3d([0, 1])

        ob = viz_3d.Ax3DPose(ax)
        ob.update(xyz_gt[i], xyz_pred[i])

        ob.ax.set_axis_off()
        ob.ax.view_init(azim, evl)

    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)

    plt.savefig(save_as)
    plt.close()



plot = False
if plot:
    val_idx = np.random.randint(val_dataset.all_seqs.shape[0], size=25)
    file_path = 'pca_poses_yz'
    plot_poses(X[val_idx], X_recon[val_idx], num_images=25, azim=0, evl=-0, save_as=file_path)
    file_path = 'pca_poses_xz'
    plot_poses(X[val_idx], X_recon[val_idx], num_images=25, azim=0, evl=90, save_as=file_path)
    file_path = 'pca_poses_xy'
    plot_poses(X[val_idx], X_recon[val_idx], num_images=25, azim=90, evl=90, save_as=file_path)

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