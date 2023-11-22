from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

c1 = np.load("true.npy")
print(c1.shape)

c2 = np.load("pred.npy")
print(c2.shape)

X = np.load("Test_Data.npy")
print(X.shape)
X = PCA(n_components=2).fit_transform(X)
print(X.shape)

plt.scatter(X[:,0], X[:,1], c=c2)
plt.show()
