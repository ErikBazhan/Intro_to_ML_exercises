'''
Created on 20.06.2025

@author: Linda Schneider
'''

import numpy as np
import cv2

# do not import more modules!
# Use OpenCV only for basic image operations such as resizing and thresholding.
# Use NumPy for the bounding box computation and for centering the symbol.
# Do not use contour detection or connected components here.


def simpleAlignment(img, size=128):
    """
    Align a grayscale symbol by centering its foreground on a fixed canvas.
    """
    if img is None:
        raise ValueError("Input image must not be None.")
    copy = img.copy()
    # Step 1: Resize the input image to a fixed square size.
    # Allowed: cv2.resize.
    copy = cv2.resize(copy, (size, size))
    # Step 2: Binarize the resized image with Otsu thresholding.
    # Allowed: cv2.threshold with Otsu.
    _, img_binary = cv2.threshold(copy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Step 3: Find the bounding box of the foreground with NumPy only.
    # Hint: The symbols are dark, the background is bright.
    # Allowed: NumPy operations such as argwhere, min, max, slicing.
    # Not allowed: cv2.findContours, connectedComponents, or similar high-level localization helpers.
    # Dark pixels correspond to the symbol foreground.
    coords = np.argwhere(img_binary == 0)
    if coords.size == 0:
        raise ValueError("No foreground pixels found in the image.")
    y_min, x_min = coords.min(axis=0) # leftmost and topmost coordinates
    y_max, x_max = coords.max(axis=0) # rightmost and bottommost coordinates
    # Step 4: Crop the grayscale region of interest from the resized image.
    # Use NumPy slicing.
    copy = copy[y_min:y_max+1, x_min:x_max+1]
    # Step 5: Resize the cropped region to fit into half the canvas while keeping aspect ratio.
    # Allowed: cv2.resize.
    target = size // 2
    h, w = copy.shape
    scale = min(target / float(h), target / float(w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    copy = cv2.resize(copy, (new_w, new_h))
    # Step 6: Place the resized symbol in the center of a blank canvas.
    # Use NumPy indexing and array assignment for centering.
    canvas = np.full((size, size), 0, dtype=np.uint8) # white canvas
    y_offset = (size - copy.shape[0]) // 2
    x_offset = (size - copy.shape[1]) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = copy
    return canvas

