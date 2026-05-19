from PIL import Image
import numpy as np


def make_kernel(ksize, sigma):
    kernel = np.zeros((ksize, ksize))
    c = ksize // 2 # center of the kernel
    for y in range(ksize):
        for x in range(ksize):
            dy = y - c
            dx = x - c
            kernel[y][x] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(dx*dx + dy*dy)/(2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)

    return  kernel


def slow_convolve(arr, k):
    image = arr.copy()  # new image
    h, w = image.shape[:2]

    U, V = k.shape
    center_y = int(np.floor(U / 2))
    center_x = int(np.floor(V / 2))
    start_y = (U - 1) // 2
    start_x = (V - 1) // 2

    for i in range(h):
        for j in range(w):
            m = 0

            for u in range(-center_y, int(np.ceil(U / 2))):
                for v in range(-center_x, int(np.ceil(V / 2))):
                    y = i - u + start_y - center_y
                    x = j - v + start_x - center_x

                    if 0 <= y < h and 0 <= x < w:
                        m += k[u + center_y, v + center_x] * arr[y, x]

            image[i, j] = m

    return image

if __name__ == '__main__':
    k = make_kernel(5, 1.0)
    
    # TODO: chose the image you prefer
    im = np.array(Image.open('data/input1.jpg')).astype(float)
    # im = np.array(Image.open('input2.jpg'))
    # im = np.array(Image.open('input3.jpg'))
    
    blurred = slow_convolve(im, k)
    detail = im - blurred
    sharpened = im + detail
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    Image.fromarray(sharpened).save('data/sharpened.jpg')
