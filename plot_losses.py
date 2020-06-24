import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

name = sys.argv[0]

def plot_psnr(path_to_file, label):
  data = pd.read_csv(path_to_file)
  #data.set_index('iteration')
  iteration = np.array(data['iteration'])
  loss = np.array(data['loss'])
  joint_loss = np.array(data['joint_loss'])
  xentropy = np.array(data['xentropy'])
  latent_loss = np.array(data['latent_loss'])

  #plt.errorbar(epochs, psnr, yerr=std, fmt='-o', label=label)
  #plt.plot(iteration, latent_loss, label='latent')
  #plt.plot(iteration[0:500], xentropy[0:500], label='entropy')
  #plt.plot(iteration[500:11000], xentropy[500:11000], label='entropy')
  #plt.plot(iteration, joint_loss, label='joint_loss')
  #plt.plot(iteration[500:2000], latent_loss[500:2000], label='latent')
  #plt.plot(iteration[500:2000], xentropy[500:2000], label='entropy')
  #plt.plot(iteration[500:2000], joint_loss[500:2000], label='joint_loss')
  plt.plot(iteration[2000:11000], joint_loss[2000:11000], label=label)
  #plt.plot(iteration[0:2000], joint_loss[0:2000], label=label)
  #plt.plot(iteration[2000:5000], joint_loss[2000:5000], label=label)

print("Number of datasets: ", len(sys.argv)-1)
for i in range(1, len(sys.argv)):
  dataset = sys.argv[i]
  plot_psnr(dataset, dataset)

plt.xlabel("Number of Iterations")
plt.ylabel("Loss")
plt.title("Training Curves")
plt.legend()
plt.savefig('training_curves.png')
