import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from fontTools.misc.bezierTools import epsilon


def load_image(path: str) -> np.ndarray:
    loaded_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if loaded_image is None:
        raise FileNotFoundError(f"Cannot load image at {path}")
    return loaded_image


def compute_histogram(image: np.ndarray) -> np.ndarray:
    """Compute a grayscale histogram with 256 bins."""
    # ToDo: Implement actual histogram calculation.
    histogram = np.zeros(256, dtype=int)
    for p in image.flatten():
        histogram[p] += 1
    return histogram


def p_helper(prob: np.ndarray, theta: int) -> tuple[float, float]:
    """Compute class probabilities p0 and p1 for threshold theta."""
    # ToDo: Implement actual probability computation.
    p0 = np.sum(prob[:theta + 1])
    p1 = np.sum(prob[theta + 1:])
    return p0, p1


def mu_helper(prob: np.ndarray, theta: int, p0: float, p1: float) -> tuple[float, float]:
    """Compute class means mu0 and mu1 for threshold theta."""
    # ToDo: Implement actual mean computation.
    mu0 = 0.0
    mu1 = 0.0
    for i in range(0, theta + 1):
        mu0 += prob[i] * i
    for i in range(theta + 1, len(prob)):
        mu1 += prob[i] * i
    if p0 != 0:
        mu0 /= p0
    if p1 != 0:
        mu1 /= p1

    return mu0, mu1


def otsu_threshold(histogram: np.ndarray) -> int:
    """Compute Otsu's threshold from a histogram."""
    # ToDo: Implement full Otsu algorithm.
    prob = histogram.astype(np.float64)  # later normalize
    max_variance = 0.0
    best_threshold = 0

    for theta in range(0, 255):
        p0, p1 = p_helper(prob, theta)
        mu0, mu1 = mu_helper(prob, theta, p0, p1)
        mu_total = p0 * mu0 + p1 * mu1
        ep = p0 * (mu0 - mu_total)**2 + p1 * (mu1 - mu_total)**2
        if ep > max_variance:
            max_variance = ep
            best_threshold = theta

    return int(best_threshold)


def otsu_binarize(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Binarize an image using Otsu's threshold."""
    # ToDo: Combine the helper functions to produce the binarized image.
    theta = otsu_threshold(compute_histogram(image))
    binarized = (image > theta).astype(np.uint8) * 255

    return binarized, theta


def custom_binarization(image: np.ndarray, theta: int) -> tuple[np.ndarray, int]:
    # ToDo: Binarize the image with a custom value.
    new_image = np.where(image > theta, 255, 0).astype(np.uint8)
    return new_image, theta


# Compatibility aliases for unit tests
create_greyscale_histogram = compute_histogram
calculate_otsu_threshold = otsu_threshold

def otsu(image: np.ndarray) -> np.ndarray:
    """Return binarized image using Otsu's method (wrapper for tests)."""
    bin_img, _ = otsu_binarize(image)
    return bin_img

def binarize_threshold(image: np.ndarray, theta: int) -> np.ndarray:
    """Simple thresholding utility used in tests."""
    return np.where(image > theta, 255, 0).astype(np.uint8)

if __name__ == '__main__':
    # Load grayscale image.
    base_dir = Path(__file__).resolve().parent
    loaded_image = load_image(str(base_dir / 'data' / 'runes.png'))

    # Compute Otsu's binarization or perform a custom binarization. Comment out one of the options.
    # binarized_image, threshold = otsu_binarize(loaded_image)
    binarized_image, threshold = custom_binarization(loaded_image, 180)

    # Display the original and the binarized image next to each other.
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(loaded_image, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    if binarized_image.size != 0:
        plt.subplot(1, 2, 2)
        plt.imshow(binarized_image, cmap='gray')
        plt.title(f"Otsu Binarization (t={threshold})")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
