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
    img_f = img.astype(np.float32)
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)  # x-derivative
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)  # y-derivative

    # Compute magnitude and orientation
    magnitude = np.hypot(gx, gy).astype(np.float32)
    # Unsigned orientation for HOG in [0, 180).
    orientation = np.rad2deg(np.arctan2(gy, gx)).astype(np.float32)
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
    
    # Dalal & Triggs, HOG §6.3 (p.4): split the window into regular spatial cells so each cell
    # accumulates a local gradient-orientation histogram; this keeps the descriptor dense and local.
    height, width = magnitude.shape
    num_cells_y = height // cell_size
    num_cells_x = width // cell_size
    valid_height = num_cells_y * cell_size
    valid_width = num_cells_x * cell_size
    histograms = np.zeros((num_cells_y, num_cells_x, num_bins), dtype=np.float32)
    # use unsigned orientations in [0, 180) and 9 evenly spaced bins by default.
    bin_size = 180.0 / num_bins  # Each bin covers this many degrees
    for y in range(valid_height):
        for x in range(valid_width):
            mag = float(magnitude[y, x])
            angle = float(orientation[y, x])

            # bilinear voting across neighboring orientation bins reduces aliasing.
            bin_pos = angle / bin_size
            lower_bin = int(np.floor(bin_pos)) % num_bins
            upper_bin = (lower_bin + 1) % num_bins
            upper_bin_weight = bin_pos - np.floor(bin_pos)
            lower_bin_weight = 1.0 - upper_bin_weight

            # bilinear voting across neighboring cells reduces sensitivity to small shifts.
            cell_y = (y + 0.5) / cell_size - 0.5
            cell_x = (x + 0.5) / cell_size - 0.5
            y0 = int(np.floor(cell_y))
            x0 = int(np.floor(cell_x))
            y1 = y0 + 1
            x1 = x0 + 1
            wy1 = cell_y - y0
            wx1 = cell_x - x0
            wy0 = 1.0 - wy1
            wx0 = 1.0 - wx1

            # Vote the gradient magnitude into the 4 surrounding cell/bin combinations.
            for cy, wy in ((y0, wy0), (y1, wy1)):
                if cy < 0 or cy >= num_cells_y:
                    continue
                for cx, wx in ((x0, wx0), (x1, wx1)):
                    if cx < 0 or cx >= num_cells_x:
                        continue
                    spatial_weight = wy * wx
                    vote = mag * spatial_weight
                    histograms[cy, cx, lower_bin] += vote * lower_bin_weight
                    histograms[cy, cx, upper_bin] += vote * upper_bin_weight
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

    block_histograms = []
    # Paper site 6 bottom: Block Normalization schemes.
    for y in range(n_cells_y - block_size + 1):
        for x in range(n_cells_x - block_size + 1):
            block = cell_hist[y:y+block_size, x:x+block_size, :].flatten()
            # in the paper 4 differnt normalization schemes are tests,
            # 3 performed equal:  L2-Hys, L2-norm and L1-sqrt; we used L2-Hys
            # L2-Hys: L2 norm followed by clipping followed by renormalizing
            norm = np.sqrt(np.sum(block * block) + eps * eps) # first norm
            block = block / norm
            block = np.clip(block, 0.0, 0.2) # clipping
            norm = np.sqrt(np.sum(block * block) + eps * eps) # renormalizing
            block_histograms.append(block / norm)
    if not block_histograms:
        return np.array([], dtype=np.float32)
    return np.concatenate(block_histograms).astype(np.float32)

