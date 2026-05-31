from PIL import Image
import numpy as np


def make_kernel(ksize, sigma):
    # Create a Gaussian kernel of size ksize x ksize
    kernel = np.zeros((ksize, ksize), dtype=np.float64)
    center = (ksize - 1) / 2.0

    for i in range(ksize):
        for j in range(ksize):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))

    # Normalize so that the kernel sums to 1
    kernel = kernel / kernel.sum()
    return kernel


def slow_convolve(arr, k):
    # Convert input to float for safe computation
    arr = np.asarray(arr, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)

    # True convolution uses a flipped kernel
    k = k[::-1, ::-1]

    kh, kw = k.shape
    pad_top = kh // 2
    pad_bottom = (kh - 1) // 2
    pad_left = kw // 2
    pad_right = (kw - 1) // 2

    # pad_top = (kh - 1) // 2  # For round numbers: floor
    # pad_bottom = kh // 2 # For odd numbers: floor
    # pad_left = (kw - 1) // 2 # For odd numbers: ceiling
    # pad_right = kw // 2 # For round numbers: ceiling

    # Handle grayscale images
    if arr.ndim == 2:
        h, w = arr.shape

        # Zero padding
        padded = np.zeros((h + pad_top + pad_bottom, w + pad_left + pad_right), dtype=np.float64)
        padded[pad_top:pad_top + h, pad_left:pad_left + w] = arr

        out = np.zeros((h, w), dtype=np.float64)

        for i in range(h):
            for j in range(w):
                value = 0.0
                for u in range(kh):
                    for v in range(kw):
                        value += k[u, v] * padded[i + u, j + v]
                out[i, j] = value

        return out

    # Handle RGB images
    if arr.ndim == 3:
        h, w, c = arr.shape
        out = np.zeros((h, w, c), dtype=np.float64)

        for ch in range(c):
            channel = arr[:, :, ch]
            padded = np.zeros((h + pad_top + pad_bottom, w + pad_left + pad_right), dtype=np.float64)
            padded[pad_top:pad_top + h, pad_left:pad_left + w] = channel

            for i in range(h):
                for j in range(w):
                    value = 0.0
                    for u in range(kh):
                        for v in range(kw):
                            value += k[u, v] * padded[i + u, j + v]
                    out[i, j, ch] = value

        return out

    raise ValueError("Input array must be either 2D or 3D.")


if __name__ == '__main__':
    k = make_kernel(5, 3)   # todo: find better parameters
    
    # TODO: chose the image you prefer
    im = np.array(Image.open('data/input1.jpg'))
    # im = np.array(Image.open('data/input2.jpg'))
    # im = np.array(Image.open('data/input3.jpg'))

    
    
    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result
    blurred = slow_convolve(im, k)

    # Unsharp masking:
    # result = input + (input - blurred)
    result = im.astype(np.float64) + (im.astype(np.float64) - blurred)

    # Clip to valid range and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Save the result
    out = Image.fromarray(result)
    out.save('output.png')
