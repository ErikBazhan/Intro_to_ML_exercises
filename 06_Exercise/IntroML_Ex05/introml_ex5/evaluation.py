import numpy as np


def confusion_matrix(y_true, y_pred):
    """
    Compute a confusion matrix using NumPy only.

    Requirements:
        - convert both inputs to NumPy arrays
        - check that they have the same one-dimensional shape
        - infer the number of classes from the largest observed label
        - count how often every pair (true_label, pred_label) occurs
        - return an integer matrix of shape (num_classes, num_classes)
    """
    # Convert inputs to NumPy arrays.
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Validate the shapes.
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Both inputs must be one-dimensional arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError("Both inputs must have the same length.")

    # Handle the empty-input case.
    if len(y_true) == 0:
        return np.zeros((0, 0), dtype=int)
    

    # Infer the number of classes.
    num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    # Initialize the confusion matrix.
    cm = np.zeros((num_classes, num_classes), dtype=int)
    # Fill the confusion matrix.
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    return cm
# Euclidean and cosine distance behaved similarly, but Euclidean was slightly better.
# NBNN performed noticeably better than KNN by about 4–5 percentage points.
# Logistic Regression worked best, with 90.39% accuracy. But takes signficantly longer to train than the other methods (over a Minute vs seconds).