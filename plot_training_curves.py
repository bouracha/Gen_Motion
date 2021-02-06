import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

name = sys.argv[0]

def plot_losses(path_to_file, label, y='val_loss', x='Epoch'):
  data = pd.read_csv("plot_data/"+str(path_to_file)+".csv")

  epoch = np.array(data[x])
  reconstruction = np.array(data[y])
  #KL = np.array(data['val_KL'])

  thermalised_history = 10
  print("{} thermalised mean: {}".format(label, np.mean(reconstruction[-thermalised_history:])))
  print("{} thermalised STD: {}".format(label, np.std(reconstruction[-thermalised_history:])))

  if label[:2] == "ae":
    plt.plot(epoch, reconstruction, label=label, linestyle='--')
  else:
    plt.plot(epoch, reconstruction, label=label)
    #plt.plot(epoch, KL, label=label+"_KL")

print("Number of datasets: ", len(sys.argv)-1)
#for i in range(1, len(sys.argv)):
  #dataset = sys.argv[i]
  #plot_losses(path_to_file=dataset, label=dataset)
#plot_losses(path_to_file="losses", label="train MSE", y='train_reconstruction')
plot_losses(path_to_file="losses", label="val MSE", y='val_reconstruction')


#x = np.linspace(0, 50, 10)
#plt.plot(x, x*0+3.5, label="Threshold for optimum reconstruction", linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Validation Curves")
plt.legend()
plt.savefig('training_curves.png')