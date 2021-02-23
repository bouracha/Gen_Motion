import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

name = sys.argv[0]

def plot_losses(path_to_file, label, y='val_loss', x='Epoch', errorbar=None, need_thermalisation=False, color='b', linestyle='-'):
  #data = pd.read_csv("plot_data/"+str(path_to_file)+".csv")
  data = pd.read_csv(str(path_to_file))

  x_plot = np.array(data[x])
  y_plot = np.array(data[y])

  if need_thermalisation:
    thermalised_history = 10
    print("{} thermalised mean: {}".format(label, np.mean(y_plot[-thermalised_history:])))
    print("{} thermalised STD: {}".format(label, np.std(y_plot[-thermalised_history:])))

  if errorbar is None:
    plt.plot(x_plot, y_plot, label=label, linestyle=linestyle)
  else:
    yerr = np.array(data[errorbar])
    plt.errorbar(x_plot[:-4], y_plot[:-4], yerr=yerr[:-4], label=label, color=color, linestyle=linestyle)

n_z = 2
file_name="added_noise"
plot_losses(path_to_file="pca/"+str(file_name)+'_'+str(n_z)+".csv", label="PCA", y='MSE', x='num_occlusions', errorbar='STD', color='y')
plot_losses(path_to_file="deep_"+str(n_z)+"_wd_AE/"+str(file_name)+".csv", label="AE", y='MSE', x='num_occlusions', errorbar='STD', color='b')
plot_losses(path_to_file="deep_"+str(n_z)+"_beta_VAE/"+str(file_name)+".csv", label="VAE ($\sigma^2_{recon}=[1]$)", y='MSE', x='num_occlusions', errorbar='STD', color='r')
plot_losses(path_to_file="deep_"+str(n_z)+"_VAE/"+str(file_name)+".csv", label="VAE", y='MSE', x='num_occlusions', errorbar='STD', color='g')

plt.yscale('log')
plt.xscale('log')
x = np.linspace(0, 96, 10)
#plt.plot(x, x*0+0.55882453918457, label="", linestyle='--', color='y')
#plt.plot(x, x*0+0.427002441354359, label="", linestyle='--', color='b')
#plt.plot(x, x*0+0.406370745100563, label="", linestyle='--', color='r')
#plt.plot(x, x*0+0.527691416260059, label="", linestyle='--', color='g')
plt.xlabel("Scale of Gaussian Noise Added")
plt.ylabel("MSE reconstruction")
plt.title("Validation Curves")
plt.legend(prop={'size': 15})
plt.savefig('training_curves.png')