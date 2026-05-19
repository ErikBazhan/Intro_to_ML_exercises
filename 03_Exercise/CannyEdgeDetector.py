import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import convo

#
# NO MORE MODULES ALLOWED
#


def gaussFilter(img_in, ksize, sigma):
    """
    filter the image with a gauss kernel
    :param img_in: 2D greyscale image (np.ndarray)
    :param ksize: kernel size (int)
    :param sigma: sigma (float)
    :return: (kernel, filtered) kernel and gaussian filtered image (both np.ndarray)
    """
    kernel = convo.make_kernel(ksize, sigma)
    filtered = convolve(img_in, kernel).astype(int)
    return kernel, filtered


def sobel(img_in):
    """
    applies the sobel filters to the input image
    Watch out! scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    """
    sx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    sy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

    gx = convolve(img_in, sx, mode='constant', cval=0).astype(int)
    gy = convolve(img_in, sy, mode='constant', cval=0).astype(int)

    return gx, gy



def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    g = np.sqrt(gx**2 + gy**2).astype(int)
    theta = np.arctan2(gy, gx)
    return g, theta


def convertAngle(angle):
    """
    compute nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
    angle = np.rad2deg(angle) % 180
    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        return 0
    elif (22.5 <= angle < 67.5):
        return 45
    elif (67.5 <= angle < 112.5):
        return 90
    else:
        return 135


def maxSuppress(g, theta):
    """
    calculate maximum suppression
    :param g:  (np.ndarray)
    :param theta: 2d image (np.ndarray)
    :return: max_sup (np.ndarray)
    """

    max_sup = np.zeros(g.shape, dtype=g.dtype)

    for y in range(1, theta.shape[0] - 1):
        for x in range(1, theta.shape[1] - 1):
            angle = convertAngle(theta[y, x])

            if angle == 0:
                before = g[y, x - 1]
                after = g[y, x + 1]
            elif angle == 45:
                before = g[y + 1, x - 1]
                after = g[y - 1, x + 1]
            elif angle == 90:
                before = g[y - 1, x]
                after = g[y + 1, x]
            else: # 135
                before = g[y - 1, x - 1]
                after = g[y + 1, x + 1]

            if g[y, x] >= before and g[y, x] >= after:
                max_sup[y, x] = g[y, x]

    return max_sup



def hysteris(max_sup, t_low, t_high):
    """
    calculate hysteris thresholding.
    Attention! This is a simplified version of the lectures hysteresis.
    Please refer to the definition in the instruction

    :param max_sup: 2d image (np.ndarray)
    :param t_low: (int)
    :param t_high: (int)
    :return: hysteris thresholded image (np.ndarray)
    """
    threshimg = np.zeros(max_sup.shape, dtype=int)

    threshimg[max_sup <= t_low] = 0
    threshimg[(max_sup > t_low) & (max_sup <= t_high)] = 1
    threshimg[max_sup > t_high] = 2

    result = np.zeros_like(max_sup, dtype=int)

    for y in range(max_sup.shape[0]):
        for x in range(max_sup.shape[1]):
            if threshimg[y, x] == 2:
                result[y, x] = 255
                # Check 8-connected neighbors
                for ny in range(max(0, y - 1), min(max_sup.shape[0], y + 2)): # Loop through neighboring rows
                    for nx in range(max(0, x - 1), min(max_sup.shape[1], x + 2)): # Loop through neighboring columns
                        if ny == y and nx == x: # Skip the current pixel
                            continue
                        if threshimg[ny, nx] == 1: # If the neighbor is a weak edge, mark it as a strong edge
                            result[ny, nx] = 255

    return result


def canny(img):
    # gaussian
    kernel, gauss = gaussFilter(img, 5, 2)

    # sobel
    gx, gy = sobel(gauss)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()
    plt.show()

    # gradient directions
    g, theta = gradientAndDirection(gx, gy)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()

    # maximum suppression
    maxS_img = maxSuppress(g, theta)

    # plotting
    plt.imshow(maxS_img, 'gray')
    plt.show()

    result = hysteris(maxS_img, 50, 75)

    return result
