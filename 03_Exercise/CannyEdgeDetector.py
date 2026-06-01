import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from PIL import Image # ONLY USED FOR TESTING 

from convo import make_kernel # IMPORTED OWN IMPLEMENTATION
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
    # TODO
    # Create a Gaussian kernel of size ksize x ksize
    kernel = make_kernel(ksize, sigma)

    # Convert input to float for safe computation
    arr = np.asarray(img_in, dtype=np.float64)
    filtered = convolve(arr, kernel, mode="reflect").astype(int) # reflecting edge values of image instead of zero padding

    return kernel, filtered


def sobel(img_in):
    """
    applies the sobel filters to the input image
    Watch out! scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    """
    # TODO
    img = np.asarray(img_in, dtype=np.float64)

    sobel_x = np.array( # first order derivative
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ],
        dtype=float,
    )

    sobel_y = np.array(
        [
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1],
        ],
        dtype=float,
    )

    gx = convolve(img, sobel_x, mode="reflect").astype(int)
    gy = convolve(img, sobel_y, mode="reflect").astype(int)

    return gx, gy


def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    # TODO
    g = np.hypot(gx, gy).astype(int) # gradient magnitude
    theta = np.arctan2(gy, gx) # gradient direction
    return g, theta


def convertAngle(angle):
    """
    compute nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
    # TODO
    angle = angle * 180/np.pi # angle important for knowing which neighbors to compare
    angle = np.asarray(angle, dtype=float) % 180 # normalise to 180 deg

    if angle.ndim == 0:
        a = float(angle)
        if a < 22.5 or a >= 157.5:
            return 0
        elif a < 67.5:
            return 45
        elif a < 112.5:
            return 90
        else:
            return 135

    result = np.zeros_like(angle, dtype=int) # result = 0 (if other cases dont match its just 0)
    result[(angle >= 22.5) & (angle < 67.5)] = 45 # (0+45)/2 = 22.5; 45 + 22.5 = 67.5
    result[(angle >= 67.5) & (angle < 112.5)] = 90 # (45+90)/2 = 67.5; 67.5 + 22.5 = 112.5
    result[(angle >= 112.5) & (angle < 157.5)] = 135 # (90+135)/2 = 112.5; 112.5 + 22.5 = 157.5
    return result


def maxSuppress(g, theta):
    """
    calculate maximum suppression
    :param g:  (np.ndarray) (gradient strength)
    :param theta: 2d image (np.ndarray) (gradient direction)
    :return: max_sup (np.ndarray)
    """
    # TODO Hint: For 2.3.1 and 2 use the helper method above
    theta_q = convertAngle(theta)
    max_sup = np.zeros_like(g)

    h, w = g.shape # check whether pixel along gradient direction is maximum

    for i in range(1, h - 1): # for every pixel
        for j in range(1, w - 1):
            direction = theta_q[i, j]
            val = g[i, j]

            if direction == 0: # horizontal comparison (left & right)
                n1 = g[i, j - 1]
                n2 = g[i, j + 1]
            elif direction == 45: # diagonal comparison 
                n1 = g[i - 1, j + 1]
                n2 = g[i + 1, j - 1]
            elif direction == 90: # vertical comparison (upper & lower)
                n1 = g[i - 1, j]
                n2 = g[i + 1, j]
            else:  # 135 # diagonal comparison
                n1 = g[i - 1, j - 1]
                n2 = g[i + 1, j + 1]

            if val >= n1 and val >= n2: # is it really maximum compared to neighbors?
                max_sup[i, j] = val # if yes keep value

    return max_sup


def hysteris(max_sup, t_low, t_high):
    """
    calculate hysteris thresholding.
    Attention! This is a simplified version of the lectures hysteresis.
    Please refer to the definition in the instruction

    :param max_sup: 2d image (np.ndarray)
    :param t_low: (int) (lower threshold)
    :param t_high: (int) (higher threshold)
    :return: hysteris thresholded image (np.ndarray)
    """
    # TODO
    max_sup = np.asarray(max_sup)
    h, w = max_sup.shape

    strong = max_sup >= t_high # pixel with strong gradient
    weak = (max_sup >= t_low) & ~strong # pixel with gradient between low & high

    result = np.zeros_like(max_sup, dtype=np.uint8)
    result[strong] = 255

    for i in range(1, h - 1): # keep only weak if it contains stron in its neighborhood
        for j in range(1, w - 1):
            if weak[i, j]:
                if np.any(strong[i - 1:i + 2, j - 1:j + 2]):
                    result[i, j] = 255

    return result


def canny(img):
    # gaussian
    kernel, gauss = gaussFilter(img, 5, 2)

    # sobel
    gx, gy = sobel(gauss) # calculate sobel with gaussian image

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

if __name__ == '__main__':
    im = Image.open('data/input1.jpg').convert("L")
    result = canny(im)
    out = Image.fromarray(result)
    out.save('output_canny.png')
