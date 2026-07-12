import numpy as np


class NBNNClassifier:
    def __init__(self, metric="euclidean"):
        # Distance metric: "euclidean" or "cosine".
        self.metric = metric
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Store training data and labels as NumPy arrays.

        Requirements:
            - convert X and y to NumPy arrays
            - validate shapes
            - store the sorted unique class labels in self.classes_
            - return self
        """
        # Convert X and y to NumPy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        # Validate shapes
        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        # Store
        self.X_train = X
        self.y_train = y
        self.classes_ = np.sort(np.unique(y))
        return self

    def _euclidean_distances(self, x):
        """Return the Euclidean distance from x to all training samples."""
        return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

    def _cosine_distances(self, x):
        """
        Return the cosine distance from x to all training samples.

        Use the same convention as in knn.py:
            cosine_distance = 1 - cosine_similarity
        """
        dot = np.dot(self.X_train, x)
        norm_x = np.linalg.norm(x)
        norm_X_train = np.linalg.norm(self.X_train, axis=1)
        # Avoid division by zero
        norm_x = np.maximum(norm_x, 1e-10)
        norm_X_train = np.maximum(norm_X_train, 1e-10)
        cosine_similarity = dot / (norm_X_train * norm_x)
        return 1 - cosine_similarity


    def _class_scores(self, distances):
        """
        Compute one score per class.

        For each class, use the distance of the nearest training sample from
        that class. The predicted class is the class with the smallest score.
        """
        scores = np.full(len(self.classes_), np.inf)
        for i, c in enumerate(self.classes_):
            class_distances = distances[self.y_train == c]
            if len(class_distances) > 0:
                scores[i] = np.min(class_distances)
        return scores

    def predict(self, X):
        """
        Predict labels for one or more samples with the NBNN rule.

        Requirements:
            - allow either a single sample or a batch
            - compute distances to all training samples
            - convert them into class-wise scores
            - return the class label with the smallest score
        """
        X = np.asarray(X)
        if self.X_train is None or self.y_train is None:
            raise ValueError("NBNNClassifier must be fitted before calling predict")

        if X.ndim == 1:
            X = X.reshape(1, -1)
        y_pred = []
        for x in X:
            if self.metric == "euclidean":
                distances = self._euclidean_distances(x)
            elif self.metric == "cosine":
                distances = self._cosine_distances(x)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
            scores = self._class_scores(distances)
            y_pred.append(self.classes_[np.argmin(scores)])
        return np.array(y_pred)
