import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def load_image(path: str) -> np.ndarray:
    loaded_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if loaded_image is None:
        raise FileNotFoundError(f"Cannot load image at {path}")
    return loaded_image


def compute_histogram(image: np.ndarray) -> np.ndarray:
    """Compute a grayscale histogram with 256 bins."""
    # ToDo: Implement actual histogram calculation.
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    return histogram


def p_helper(prob: np.ndarray, theta: int) -> tuple[float, float]: # Input probability must be from normalized histogram!
    """Compute class probabilities p0 and p1 for threshold theta."""
    # ToDo: Implement actual probability computation.
    p0 = float(np.sum(prob[: theta + 1])) # Sum of class 0 appearance in the histogram
    p1 = float(np.sum(prob[theta + 1 :])) # Sum of class 1 appearance in the histogram (p0 - 1)
    return p0, p1


def mu_helper(prob: np.ndarray, theta: int, p0: float, p1: float) -> tuple[float, float]:
    """Compute class means mu0 and mu1 for threshold theta."""
    # ToDo: Implement actual mean computation.
    values = np.arange(len(prob))

    if p0 == 0 or p1 == 0:
        return 0.0, 0.0

    mu0 = float(np.sum(values[: theta + 1] * prob[: theta + 1]) / p0)
    mu1 = float(np.sum(values[theta + 1 :] * prob[theta + 1 :]) / p1)

    return mu0, mu1


def otsu_threshold(histogram: np.ndarray) -> int:
    """Compute Otsu's threshold from a histogram."""
    # ToDo: Implement full Otsu algorithm.
    prob = histogram.astype(np.float64) 
    prob /= prob.sum() 

    max_variance = -1.0
    best_threshold = 0

    for theta in range(len(prob)):
        p0, p1 = p_helper(prob, theta)

        if p0 == 0.0 or p1 == 0.0:
            continue

        mu0, mu1 = mu_helper(prob, theta, p0, p1)

        between_class_variance = p0 * p1 * (mu0 - mu1) ** 2

        if between_class_variance > max_variance:
            max_variance = between_class_variance
            best_threshold = theta

    return int(best_threshold)


def otsu_binarize(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Binarize an image using Otsu's threshold."""
    # ToDo: Combine the helper functions to produce the binarized image.
    histogram = compute_histogram(image)
    theta = otsu_threshold(histogram)
    binarized = np.where(image > theta, 255, 0).astype(np.uint8)
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
