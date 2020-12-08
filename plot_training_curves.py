import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

name = sys.argv[0]

def plot_losses(path_to_file, label):
  data = pd.read_csv(path_to_file)

  epoch = np.array(data['Epoch'])
  reconstruction = np.array(data['val_reconstruction'])
  #KL = np.array(data['val_KL'])

  plt.plot(epoch, reconstruction, label=label)
  #plt.plot(epoch, KL, label=label+"_KL")

print("Number of datasets: ", len(sys.argv)-1)
for i in range(1, len(sys.argv)):
  dataset = sys.argv[i]
  plot_losses(dataset, dataset)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Validation Curves")
plt.legend()
plt.savefig('training_curves.png')