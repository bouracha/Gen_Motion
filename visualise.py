import umap.umap_ as umap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np


train_data = pd.read_csv('latents/train_all.csv')

print(train_data.shape)
#print(train_data[0:100].shape)

standardise = StandardScaler()
reducer = umap.UMAP()
train_data_scaled = standardise.fit_transform(train_data)
embedding = reducer.fit_transform(train_data_scaled)

print(embedding.shape)

plt.figure(figsize=(20, 20))
plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.1, s=3)

jumping_data = pd.read_csv('latents/running_train_z.csv')
print(jumping_data.shape)
#jumping_data = np.reshape(jumping_data, (len(jumping_data), 512))
#print(jumping_data.shape)
jumping_data_scaled = np.asanyarray(standardise.transform(jumping_data))
print(jumping_data.shape)
print(len(jumping_data))
jumping_embedding = reducer.transform(jumping_data)
print("train data transfromed: ", jumping_embedding.shape)
plt.scatter(jumping_embedding[:, 0], jumping_embedding[:, 1], color="r", alpha=1.0, s=3)


plt.title("Behaviour Embedding")
#plt.legend()
plt.savefig('embedding.png')