import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

name = sys.argv[0]

def plot_losses(path_to_file, label, y='val_loss', x='Epoch', errorbar=None, need_thermalisation=False, color='b', linestyle='-', y2=None, y2_label='KL'):
  fontsize = 10
  #data = pd.read_csv("plot_data/"+str(path_to_file)+".csv")
  data = pd.read_csv(str(path_to_file))[:300]

  fig, ax1 = plt.subplots()

  x_plot = np.array(data[x])
  y_plot = -np.array(data[y])

  #set max value
  #for i in range(50, len(y_plot)):
  #  if y_plot[i] < -150:
  #    y_plot[i] = y_plot[i-1]
  #y_plot[y_plot < -150] = -150

  if need_thermalisation:
    thermalised_history = 10
    print("{} thermalised mean: {}".format(label, np.mean(y_plot[-thermalised_history:])))
    print("{} thermalised STD: {}".format(label, np.std(y_plot[-thermalised_history:])))

  if errorbar is None:
    ax1.plot(x_plot, y_plot, label=label, color=color, linestyle=linestyle)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel("VLB")
  else:
    yerr = np.array(data[errorbar])
    plt.errorbar(x_plot, y_plot, yerr=yerr, label=label, color=color, linestyle=linestyle)

  if y2 is not None:
    ax2 = ax1.twinx()
    y2_plot = np.array(data[y2])
    ax2.plot(x_plot, y2_plot, label=y2_label, color='r', linestyle=linestyle)
    ax2.set_ylabel('KL')

  x = np.linspace(25, 65, 10)
  ax2.plot(x * 0 + 20, x, label=r"$ \beta =0.001$", linestyle='--', color='r')

  plt.xlabel("Epoch", fontsize=fontsize)
  # plt.xlabel("Scale of Gaussian Noise Added", fontsize=fontsize)
  # plt.ylabel("VLB", fontsize=fontsize)
  plt.title("", fontsize=fontsize)
  ax1.legend(prop={'size': 12}, bbox_to_anchor=(0.5, 0., 0.5, 0.5))
  ax2.legend(prop={'size': 12}, bbox_to_anchor=(0.5, 0., 0.5, 0.3))
  plt.savefig('plot.png')



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

#plot_losses(path_to_file="res_2_MNIST/losses.csv", label="VAE (2)", y="val_VLB", x="Epoch", linestyle='-')
#plot_losses(path_to_file="res_2_2_MNIST/losses.csv", label="2-layer (2 2)", y="val_VLB", x="Epoch", linestyle='-')
#plot_losses(path_to_file="res_2_2_2_MNIST/losses.csv", label="3-layer (2 2 2)", y="val_VLB", x="Epoch", linestyle='-')
#plot_losses(path_to_file="res_2_2_2_2_MNIST/losses.csv", label="4-layer (2 2 2 2)", y="val_VLB", x="Epoch", linestyle='--')
#plot_losses(path_to_file="VDVAE_2_2_MNIST/losses.csv", label="2-layer (2 2) w/o res", y="val_VLB", x="Epoch", linestyle=':')
#plot_losses(path_to_file="VDVAE_2_2_2MNIST/losses.csv", label="3-layer (2 2 2) w/o res", y="val_VLB", x="Epoch", linestyle=':')
#plot_losses(path_to_file="res_warmup_2_2_MNIST/losses.csv", label="2-layer (2 2) with warmup", y="val_VLB", x="Epoch", linestyle=':')
#plot_losses(path_to_file="res_warmup_2_2_2_MNIST/losses.csv", label="3-layer (2 2 2) with warmup", y="val_VLB", x="Epoch", linestyle=':')
#plot_losses(path_to_file="res_warmup_2_2_2_2_MNIST/losses.csv", label="4-layer (2 2 2 2) with warmup", y="val_VLB", x="Epoch", linestyle='--')
#plot_losses(path_to_file="rezero_warmup_2_2_2_2_MNIST/losses.csv", label="4-layer (2 2 2 2) ReZero & Warmup", y="val_VLB", x="Epoch", linestyle='-')
#plot_losses(path_to_file="rezero_2_2_2_2_MNIST/losses.csv", label="4-layer (2 2 2 2) ReZero", y="val_VLB", x="Epoch", linestyle='-')
#plot_losses(path_to_file="vdvae_2_MNIST_VAE/kls.csv", label="KL 0", y="val0", x="Epoch", linestyle='dotted')
#plot_losses(path_to_file="vdvae_2_MNIST_VAE/kls.csv", label="KL 1", y="val1", x="Epoch", linestyle='dotted')
#plot_losses(path_to_file="vdvae_2_MNIST_VAE/kls.csv", label="KL 2", y="val2", x="Epoch", linestyle='dotted')
#plot_losses(path_to_file="vdvae_2_MNIST_VAE/kls.csv", label="KL 3", y="val3", x="Epoch", linestyle='dotted')
#plot_losses(path_to_file="vdvae_2_MNIST_VAE/kls.csv", label="KL 4", y="val4", x="Epoch", linestyle='dotted')
#plot_losses(path_to_file="vdvae_2_MNIST_VAE/kls.csv", label="KL 5", y="val5", x="Epoch", linestyle='dotted')
#plot_losses(path_to_file="vdvae_2_MNIST_VAE/kls.csv", label="KL 6", y="val6", x="Epoch", linestyle='dotted')
#plot_losses(path_to_file="ladder_MNIST/losses.csv", label="Train", y="train_VLB", x="Epoch", linestyle='-')
#plot_losses(path_to_file="ladder_MNIST/losses.csv", label="Val", y="val_VLB", x="Epoch", linestyle='-')
#plot_losses(path_to_file="ladder_MNIST/losses.csv", label="Total KL", y="val_KL", x="Epoch", linestyle='-')
#plot_losses(path_to_file="ladder_MNIST/kls.csv", label="KL $z_0$", y="val0", x="Epoch", linestyle=':')
#plot_losses(path_to_file="ladder_MNIST/kls.csv", label="KL $z_1$", y="val1", x="Epoch", linestyle=':')
#plot_losses(path_to_file="ladder_MNIST/kls.csv", label="KL $z_2$", y="val2", x="Epoch", linestyle=':')
#plot_losses(path_to_file="ladder_MNIST/kls.csv", label="KL $z_3$", y="val3", x="Epoch", linestyle=':')
#plot_losses(path_to_file="ladder_MNIST/kls.csv", label="KL $z_4$", y="val4", x="Epoch", linestyle=':')
plot_losses(path_to_file="ladder_beta/losses.csv", label="VLB", y="val_loss", x="Epoch", linestyle='-', y2='val_KL')
#plot_losses(path_to_file="ladder_MNIST/losses.csv", label="Val", y="val_VLB", x="Epoch", linestyle='-')
#plot_losses(path_to_file="ladder_MNIST/losses.csv", label="Total KL", y="val_KL", x="Epoch", linestyle='-')

#plot_losses(path_to_file="deep_"+str(n_z)+"_beta_VAE/"+str(file_name)+".csv", label="VAE", y='MSE', x='num_samples', errorbar='STD', color='m')


#plt.yscale('log')
#plt.xscale('log')
x = np.linspace(0, 1000, 10)
#plt.plot(x, x*0+0.55882453918457, label="", linestyle='--', color='y')
#plt.plot(x, x*0+0.427002441354359, label="", linestyle='--', color='b')
#plt.plot(x, x*0+0.406370745100563, label="", linestyle='--', color='r')
#plt.plot(x, x*0+0.527691416260059, label="", linestyle='--', color='g')
#plt.plot(x*0 + 200, x, label=r"$ \beta =1.0$", linestyle='--', color='r')
