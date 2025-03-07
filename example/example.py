import numpy as np
from simple_kmeans import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate Data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Apply Custom K-Means
kmeans = KMeans(k=3)
clusters = kmeans.fit_predict(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.legend()
plt.show()
