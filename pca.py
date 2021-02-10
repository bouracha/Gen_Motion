import utils.data_utils as data_utils
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

acts = data_utils.define_actions("all")
X, _, _ = data_utils.load_data_3d("h3.6m/dataset/", [1, 6, 7, 8, 9], acts, sample_rate=2, seq_len=1)
X_val, _, _ = data_utils.load_data_3d("h3.6m/dataset/", [11], acts, sample_rate=2, seq_len=1)

print(X.shape)

pca = PCA(n_components=2)
z = pca.fit_transform(X)

print(z.shape)
print(pca.explained_variance_ratio_)

z_val = pca.transform(X_val)

print(z_val.shape)

X_recon = pca.inverse_transform(z)
X_recon_val = pca.inverse_transform(z_val)

print(X_recon.shape)
print(X_recon_val.shape)

train_mse = np.mean((X_recon - X)**2)
val_mse = np.mean((X_recon_val - X_val)**2)

print("Train reconstruction MSE:{}".format(train_mse))
print("Validation reconstruction MSE:{}".format(val_mse))

plot = False
if plot:
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