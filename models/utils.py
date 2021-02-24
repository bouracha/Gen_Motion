import numpy as np

from torchvision.utils import make_grid
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

import utils.viz_3d as viz_3d


class AccumLoss(object):
    def __init__(self):
        self.reset()

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


def plot_tensor_images(image_tensor, max_num_images=25, nrow=5, show=False, save_as=None):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:max_num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if show:
        plt.show()
    if not save_as == None:
        plt.savefig(save_as)
    plt.close()


def plot_poses(xyz_gt, xyz_pred, max_num_images=25, azim=0, evl=0, save_as=None, one_dim_grid=False):
    '''
    Function for visualizing poses: saves grid of poses.
    Assumes poses are normalised between 0 and 1
    :param xyz_gt: set of ground truth 3D joint positions (batch_size, 96)
    :param xyz_pred: set of predicted 3D joint positions (batch_size, 96)
    :param max_num_images: number of poses to plotted from given set (int)
    :param azim: azimuthal angle for viewing (int)
    :param evl: angle of elevation for viewing (int)
    :param save_as: path and name to save (str)
    '''
    xyz_gt = xyz_gt.reshape(-1, 32, 3)
    xyz_pred = xyz_pred.reshape(-1, 32, 3)

    xyz_gt = xyz_gt[:max_num_images]
    xyz_pred = xyz_pred[:max_num_images]
    num_images = xyz_gt.shape[0]

    fig = plt.figure()
    if num_images > 4:
        fig = plt.figure(figsize=(20, 20))
    elif num_images > 40:
        fig = plt.figure(figsize=(50, 50))

    if one_dim_grid:
        fig = plt.figure(figsize=(20, 4))
        grid_dim_y = 1
        grid_dim_x = num_images
    else:
        grid_dim_y = np.ceil(np.sqrt(num_images))
        grid_dim_x = grid_dim_y

    for i in range(num_images):
        ax = fig.add_subplot(grid_dim_y, grid_dim_x, i + 1, projection='3d')
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


def simulate_occlusions(X, num_occlusions=10, folder_name=""):
    '''
    Function to randomly replace num_occlusions values with their average value in the train set.
    Assumes we've save the 96 features as avg_features.csv
    :param X: set of ground truth 3D joint positions (batch_size, 96)
    :param num_occlusions: number of occlusions per pose (int)
    :param folder_name: model_path (str)
    :return: set of ground truth 3D joint positions each with num_occlusions replaced (batch_size, 96)
    '''
    m, n = X.shape
    assert (n == 96)
    X_occluded = np.copy(X)

    rng = np.random.default_rng()
    occlude_mask = np.zeros((m, n))
    occlude_mask[:, :num_occlusions] = 1.0
    occlude_mask = occlude_mask.astype('bool')
    rng.shuffle(occlude_mask, axis=1)
    assert (np.sum(occlude_mask[0]) == num_occlusions)
    assert (np.sum(occlude_mask) == num_occlusions * m)

    path = str(folder_name) + "avg_features.csv"
    avg_features = np.loadtxt(path, delimiter=',')
    avg_features_repeated = np.repeat(avg_features.reshape(1, -1), m, axis=0)
    assert (avg_features_repeated.shape == X.shape)

    X_occluded[occlude_mask] = avg_features_repeated[occlude_mask]

    return X_occluded


def add_noise(X, alpha=1.0):
    '''
    Function to add gaussian noise scaled by alpha
    :param X: set of ground truth 3D joint positions (batch_size, 96)
    :param alpha: scalinng factor for amount of noise (float)
    :return: set of ground truth 3D joint positions each with added noise (batch_size, 96)
    '''
    m, n = X.shape
    assert (n == 96)
    X_added_noise = np.copy(X)

    X_added_noise = X_added_noise + alpha * np.random.uniform(0, 1, (m, n))

    return X_added_noise


