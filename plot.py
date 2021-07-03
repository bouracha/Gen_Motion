import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

name = sys.argv[0]

def plot_losses(path_to_file, label, y='val_loss', x='Epoch', errorbar=None, need_thermalisation=False, color='b', linestyle='-'):
  #data = pd.read_csv("plot_data/"+str(path_to_file)+".csv")
  data = pd.read_csv(str(path_to_file))[:1000]

  x_plot = np.array(data[x])
  y_plot = np.array(data[y])

  #set max value
  y_plot[y_plot > 15] = 15

  if need_thermalisation:
    thermalised_history = 10
    print("{} thermalised mean: {}".format(label, np.mean(y_plot[-thermalised_history:])))
    print("{} thermalised STD: {}".format(label, np.std(y_plot[-thermalised_history:])))

  if errorbar is None:
    plt.plot(x_plot, y_plot, label=label, linestyle=linestyle)
  else:
    yerr = np.array(data[errorbar])
    plt.errorbar(x_plot, y_plot, yerr=yerr, label=label, color=color, linestyle=linestyle)

n_z = 10
file_name="kls.csv"
#plot_losses(path_to_file="pca/"+str(file_name)+'_'+str(n_z)+".csv", label="PCA", y='MSE', x='num_occlusions', errorbar='STD', color='y')
#plot_losses(path_to_file="deep_"+str(n_z)+"_wd_AE/"+str(file_name)+".csv", label="AE", y='MSE', x='num_occlusions', errorbar='STD', color='b')
#plot_losses(path_to_file="deep_"+str(n_z)+"_beta_VAE/"+str(file_name)+".csv", label="VAE ($\sigma^2_{recon}=[1]$)", y='MSE', x='num_occlusions', errorbar='STD', color='r')
#plot_losses(path_to_file="deep_"+str(n_z)+"_VAE/"+str(file_name)+".csv", label="VAE (1 sample)", y='MSE', x='alpha_noise', errorbar='STD', color='g')
#plot_losses(path_to_file="deep_"+str(n_z)+"_VAE/"+str(file_name)+"_nsamp10.csv", label="VAE (10 samples)", y='MSE', x='alpha_noise', errorbar='STD', color='m')
#plot_losses(path_to_file="classify_noisey_embeddings_no_bottleneck.csv", label="Inputs", y='inputs', x='noise_level', errorbar='inputs_std', color='m')
#plot_losses(path_to_file="classify_noisey_embeddings_no_bottleneck.csv", label="AE embeddings", y='AE', x='noise_level', errorbar='AE_std', color='b')
#plot_losses(path_to_file="classify_noisey_embeddings_no_bottleneck.csv", label="VAE ($\sigma^2_{recon}=[1]$) embeddings", y='beta_VAE', x='noise_level', errorbar='beta_VAE_std', color='r')
#plot_losses(path_to_file="classify_noisey_embeddings_no_bottleneck.csv", label="VAE embeddings", y='VAE', x='noise_level', errorbar='VAE_std', color='g')
#plot_losses(path_to_file="classify_noisey_embeddings.csv", label="VAE embedding", y='VAE', x='noise_level', errorbar='VAE_std', color='y')

plot_losses(path_to_file="vdvae_2_MNIST_VAE/kls.csv", label="KL 0", y="val0", x="Epoch", linestyle='dotted')
plot_losses(path_to_file="vdvae_2_MNIST_VAE/kls.csv", label="KL 1", y="val1", x="Epoch", linestyle='dotted')
plot_losses(path_to_file="vdvae_2_MNIST_VAE/kls.csv", label="KL 2", y="val2", x="Epoch", linestyle='dotted')
plot_losses(path_to_file="vdvae_2_MNIST_VAE/kls.csv", label="KL 3", y="val3", x="Epoch", linestyle='dotted')
plot_losses(path_to_file="vdvae_2_MNIST_VAE/kls.csv", label="KL 4", y="val4", x="Epoch", linestyle='dotted')
plot_losses(path_to_file="vdvae_2_MNIST_VAE/kls.csv", label="KL 5", y="val5", x="Epoch", linestyle='dotted')
plot_losses(path_to_file="vdvae_2_MNIST_VAE/kls.csv", label="KL 6", y="val6", x="Epoch", linestyle='dotted')

#plot_losses(path_to_file="deep_"+str(n_z)+"_beta_VAE/"+str(file_name)+".csv", label="VAE", y='MSE', x='num_samples', errorbar='STD', color='m')

fontsize=10
#plt.yscale('log')
#plt.xscale('log')
x = np.linspace(0, 96, 10)
#plt.plot(x, x*0+0.55882453918457, label="", linestyle='--', color='y')
#plt.plot(x, x*0+0.427002441354359, label="", linestyle='--', color='b')
#plt.plot(x, x*0+0.406370745100563, label="", linestyle='--', color='r')
#plt.plot(x, x*0+0.527691416260059, label="", linestyle='--', color='g')
plt.xlabel("Epoch", fontsize=fontsize)
#plt.xlabel("Scale of Gaussian Noise Added", fontsize=fontsize)
plt.ylabel("Loss", fontsize=fontsize)
plt.title("", fontsize=fontsize)
plt.legend(prop={'size': 12})
plt.savefig('plot.png')