import numpy as np

class KMeans:
    def __init__(self, k: int = 3, max_iters: int = 100, tol:float = 1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
    
    def _compute_distance(self, X, centroids):
        """
        Computes distance between each data point and the centroids using
        the appropriate metric based on dimensionality.

        Parameters:
        X (numpy array): Data points.
        centroids (numpy array): Cluster centroids.

        Returns:
        numpy array: Distance matrix.
        """

        n_dim = X.shape[1]

        if n_dim == 1:
            # Manhattan Distance
            distances  = np.abs(X[:, np.newaxis] - centroids)
        elif n_dim == 1:
            # Euclidean Distance
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        else:
            # Cosine Distance
            X_norm = np.linalg.norm(X, axis=1, keepdims=True)
            centroids_norm = np.linalg.norm(centroids, axis=1, keepdims=True)
            cosine_similarity = np.dot(X, centroids.T) / (X_norm * centroids_norm.T)
            distances = 1 - cosine_similarity
        
        return distances

    def _assign_clusters(self, X):
        """
        Assigns each data point to the closest centroid.

        Parameters:
        X (numpy array): Data points.

        Returns:
        numpy array: Cluster assignments.
        """
        distances = self._compute_distance(X, self.centroids)
        return np.argmin(distances, axis=1)

    def fit(self, X):
        """
        Fits the K-Means algorithm to the dataset.

        Parameters:
        X (numpy array): Dataset of shape (n_samples, n_features).
        """
        np.random.seed(42)
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]

        for _ in range(self.max_iters):
            clusters = self._assign_clusters(X)
            new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(self.k)]) # recalculate centroids

            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break

            self.centroids = new_centroids
    
    def predict(self, X):
        """
        Predicts cluster labels for new data points.

        Parameters:
        X (numpy array): Dataset to predict.

        Returns:
        numpy array: Cluster labels.
        """
        return self._assign_clusters(X)

    def fit_predict(self, X):
        """
        Performs both fitting and prediction.

        Parameters:
        X (numpy array): Dataset.

        Returns:
        numpy array: Cluster labels.
        """
        self.fit(X)
        return self.predict(X)

