from pathlib import Path

import numpy as np

from visualization import plot_knn_neighbors


class KNNClassifier:
    def __init__(
        self,
        n_neighbors=3,
        metric="euclidean",
        plot_neighbors=False,
        image_shape=None,
        plot_dir=None,
        plot_prefix="knn",
    ):
        # Number of nearest neighbours to use for prediction.
        self.n_neighbors = n_neighbors
        # Distance metric: "euclidean" or "cosine".
        self.metric = metric
        self.plot_neighbors = plot_neighbors
        self.image_shape = image_shape
        self.plot_dir = plot_dir
        self.plot_prefix = plot_prefix

        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store the training data and labels as NumPy arrays.

        Requirements:
            - convert X and y to NumPy arrays
            - check that X has shape (n_samples, n_features)
            - check that len(X) == len(y)
            - return self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features)")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self.X_train = X
        self.y_train = y
        return self

    def _euclidean_distances(self, x):
        """Return the Euclidean distance from x to all training samples."""
        return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

    def _cosine_distances(self, x):
        """
        Return the cosine distance from x to all training samples.

        Hint:
            cosine_distance = 1 - cosine_similarity
            cosine_similarity = (a · b) / (||a|| * ||b||)

        Make sure that zero vectors do not cause a division-by-zero error.
        """
        # Compute cosine similarity
        dot = np.dot(self.X_train, x)
        norm_x = np.linalg.norm(x)
        norm_X_train = np.linalg.norm(self.X_train, axis=1)
        # Avoid division by zero
        norm_x = np.maximum(norm_x, 1e-10)
        norm_X_train = np.maximum(norm_X_train, 1e-10)
        cosine_similarity = dot / (norm_X_train * norm_x)
        return 1 - cosine_similarity

    def _majority_vote(self, neighbor_labels):
        """
        Return the most frequent label among the nearest neighbours.

        Hint:
            np.unique(..., return_counts=True) is useful here.
            If there is a tie, choose the smallest label after sorting.
        """
        labels, counts = np.unique(neighbor_labels, return_counts=True)
        max_count = np.max(counts)
        candidates = labels[counts == max_count]
        return np.min(candidates)

    def predict(self, X):
        """
        Predict labels for one or more input samples.

        Requirements:
            - convert X to a NumPy array
            - allow a single sample with shape (n_features,)
            - compute distances with the selected metric
            - select the k nearest neighbours
            - predict by majority vote
            - optionally save neighbour plots when plot_neighbors is True
        """
        # Convert input data to a NumPy array.
        X = np.asarray(X)
        predictions = []

        if self.X_train is None or self.y_train is None:
            raise ValueError("KNNClassifier must be fitted before calling predict")

        # If a single sample is passed, turn it into shape (1, n_features).
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Iterate over each test sample to predict its label.
        for sample_index, x in enumerate(X):
            if self.metric == "euclidean":
                # Compute Euclidean distances from x to all training samples.
                distances = self._euclidean_distances(x)
            elif self.metric == "cosine":
                # Compute cosine distances from x to all training samples.
                distances = self._cosine_distances(x)
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")

            # Find the indices of the k nearest neighbours.
            nn_indices = np.argsort(distances)[: self.n_neighbors]

            # Get the labels of the nearest neighbours.
            nn_labels = self.y_train[nn_indices]

            # Find the most common label and append it to predictions.
            predicted_label = self._majority_vote(nn_labels)
            predictions.append(predicted_label)

            if self.plot_neighbors and self.image_shape is not None:
                test_image = x.reshape(self.image_shape)
                neighbor_images = [
                    self.X_train[idx].reshape(self.image_shape) for idx in nn_indices
                ]

                save_path = None
                if self.plot_dir is not None:
                    save_path = Path(self.plot_dir) / f"{self.plot_prefix}_{sample_index:02d}.png"

                plot_knn_neighbors(
                    test_image=test_image,
                    neighbor_images=neighbor_images,
                    neighbor_labels=nn_labels,
                    test_label=None,
                    pred_label=predicted_label,
                    save_path=save_path,
                )

        # Return predictions as a NumPy array.
        return np.asarray(predictions)
