'''
Histogram of Oriented Gradients utilities for exercise 3.
'''

import numpy as np
import cv2

# do not import more modules!
# You may use cv2.Sobel for the derivatives.
# Compute magnitudes, orientations, histogram binning, and block normalization yourself with NumPy.
# Do not use cv2.HOGDescriptor or any other ready-made HOG implementation.


def computeGradients(img):
    """
    Compute gradient magnitudes and unsigned orientations in degrees.
    """
    if img is None:
        raise ValueError("Input image must not be None.")

    # TODO: compute Sobel derivatives, magnitudes, and orientations.
    # Allowed: cv2.Sobel for the x/y derivatives and NumPy for the remaining computations.
    # Not allowed: any ready-made HOG or feature extraction implementation.
    if img is None:
        raise ValueError("Input image must not be None.")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.rad2deg(np.arctan2(gy, gx))

    # Convert from [-180, 180] to [0, 180)
    orientation = np.mod(orientation, 180.0)

    return magnitude, orientation


def buildCellHistograms(magnitude, orientation, cell_size=8, num_bins=9):
    """
    Accumulate orientation histograms for each cell.
    """
    if magnitude.shape != orientation.shape:
        raise ValueError("Magnitude and orientation must have the same shape.")

    # TODO: divide the image into cells and accumulate magnitudes into bins.
    # Use NumPy indexing/loops to implement the histogram accumulation yourself.
    # Do not call a library routine that directly computes cell histograms for HOG.
    if magnitude.shape != orientation.shape:
        raise ValueError("Magnitude and orientation must have the same shape.")

    h, w = magnitude.shape
    n_cells_y = h // cell_size
    n_cells_x = w // cell_size

    histograms = np.zeros((n_cells_y, n_cells_x, num_bins), dtype=np.float32)
    bin_width = 180.0 / num_bins

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y0 = cy * cell_size
            y1 = y0 + cell_size
            x0 = cx * cell_size
            x1 = x0 + cell_size

            cell_mag = magnitude[y0:y1, x0:x1].ravel()
            cell_ori = orientation[y0:y1, x0:x1].ravel()

            # Map orientations to bins
            bin_idx = np.floor(cell_ori / bin_width).astype(np.int32)
            bin_idx = np.clip(bin_idx, 0, num_bins - 1)

            # Accumulate magnitudes into bins
            for i in range(cell_mag.size):
                histograms[cy, cx, bin_idx[i]] += cell_mag[i]

    return histograms


def calculateHOG(img, cell_size=8, block_size=2, num_bins=9, eps=1e-6):
    """
    Compute a dense HOG descriptor with overlapping, normalized blocks.
    """
    # TODO: compute the final descriptor from your own cell histograms.
    # Implement the block normalization and concatenation yourself with NumPy.
    # Do not use cv2.HOGDescriptor, skimage.feature.hog, or similar helpers.
    magnitude, orientation = computeGradients(img)
    cell_hist = buildCellHistograms(
        magnitude, orientation, cell_size=cell_size, num_bins=num_bins
    )

    n_cells_y, n_cells_x, _ = cell_hist.shape

    if n_cells_y < block_size or n_cells_x < block_size:
        return np.array([], dtype=np.float32)

    hog_features = []

    for by in range(n_cells_y - block_size + 1):
        for bx in range(n_cells_x - block_size + 1):
            block = cell_hist[by:by + block_size, bx:bx + block_size, :]
            block_vec = block.ravel().astype(np.float32)

            norm = np.sqrt(np.sum(block_vec ** 2) + eps ** 2)
            block_vec = block_vec / norm

            hog_features.append(block_vec)

    return np.concatenate(hog_features).astype(np.float32)
