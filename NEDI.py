import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import sys
from PIL import Image
from helper import np2img

"""
Template used as baseline for this implementation: https://github.com/Kirstihly/Edge-Directed_Interpolation
"""

def NEDI_upscale(img, m):

    # here the upscaled image is initialized. Original pixel values are placed at even-indexed positions in the array, and new pixels are set to zero.
    h, w = img.shape
    upscaled = np.zeros((h * 2, w * 2))
    upscaled[0::2,0::2] = img

    # Neighbourhood window is set up. m is window size, which is defined when initializing the function, and is typically 4. 
    # y holds the pixel value from the low res grid
    # C holds the four neighbouring pixels that is used to predict the missing pixel value
    half_window = m // 2
    i_min = half_window + 1
    i_max = h - half_window
    j_min = half_window + 1
    j_max = w - half_window
    y = np.zeros((m**2, 1)) # This is the pixels in the window
    C = np.zeros((m**2, 4)) # C is the list of interpolation neighbour for each pixel in the window
    
    # Reconstruct the pixels at locations (2*i+1,2*j+1). This is somewhat similar to the original NEDI implementation but a bit simpler.
    # From the original approach, local covariace is estimated and used to derive interpolation weights, in this approach least-squares is used
    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            tmp = 0
            for ii in range(-half_window, half_window):
                for jj in range(-half_window, half_window):
                    base_i = i + ii
                    base_j = j + jj
                    y[tmp, 0] = upscaled[2 * base_i, 2 * base_j]
                    C[tmp, 0] = upscaled[2 * base_i-1, 2 * base_j -1 ]
                    C[tmp, 1] = upscaled[2 * base_i+1, 2 * base_j -1 ]
                    C[tmp, 2] = upscaled[2 * base_i+1, 2 * base_j +1 ]
                    C[tmp, 3] = upscaled[2 * base_i-1, 2 * base_j +1 ]
                    tmp += 1

            # interpolation weights using least squares
            a, _, _, _ = np.linalg.lstsq(C, y, rcond=None)
            
            neighbors = np.array([
                upscaled[2 * i, 2 * j],
                upscaled[2 * i + 2, 2 * j],
                upscaled[2 * i + 2, 2 * j + 2],
                upscaled[2 * i, 2 * j + 2]
            ], dtype=np.float64)
            upscaled[2 * i + 1, 2 * j + 1] = float(neighbors.dot(a).item())
    
    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            patch_idx = 0
            for di in range(-half_window, half_window):
                for dj in range(-half_window, half_window):
                    base_i = i + di
                    base_j = j + dj

                    y[patch_idx, 0] = upscaled[2 * base_i + 1, 2 * base_j]
                    C[patch_idx, 0] = upscaled[2 * base_i, 2 * base_j]
                    C[patch_idx, 1] = upscaled[2 * base_i + 1, 2 * base_j - 1]
                    C[patch_idx, 2] = upscaled[2 * base_i + 2, 2 * base_j]
                    C[patch_idx, 3] = upscaled[2 * base_i + 1, 2 * base_j + 1]
                    patch_idx += 1

            a, _, _, _ = np.linalg.lstsq(C, y, rcond=None)

            # Horizontal interpolation at (2*i+1, 2*j)
            neighbors_h = np.array([
                upscaled[2 * i, 2 * j],
                upscaled[2 * i + 1, 2 * j - 1],
                upscaled[2 * i + 2, 2 * j],
                upscaled[2 * i + 1, 2 * j + 1]
            ], dtype=np.float64)
            upscaled[2 * i + 1, 2 * j] = float(neighbors_h.dot(a).item())

            # Vertical interpolation at (2*i, 2*j+1)
            neighbors_v = np.array([
                upscaled[2 * i - 1, 2 * j + 1],
                upscaled[2 * i, 2 * j],
                upscaled[2 * i + 1, 2 * j + 1],
                upscaled[2 * i, 2 * j + 2]
            ], dtype=np.float64)
            upscaled[2 * i, 2 * j + 1] = float(neighbors_v.dot(a).item())

    # after NEDI has finished, remaining zero-valued pixels are estimated using bilinear interpolation.
    upscaled = np.clip(upscaled, 0, 255)
    bilinear = cv2.resize(img, dsize=(w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    upscaled[upscaled == 0] = bilinear[upscaled == 0]
    
    return upscaled.astype(img.dtype)

# in this function, all the channels are processed individually. 
def NEDI_predict(img, m, scale):
    arr = np.array(img).astype(np.float64)
    output_type = arr.dtype

    channels = []
    num_iterations = int(math.log(scale,2))
    for c in range(3):
        channel = arr[:, :, c]
        for _ in range(num_iterations):
            channel = NEDI_upscale(channel, m)
        current_h, current_w = channel.shape
        linear_factor = scale / (2 ** num_iterations)
        if linear_factor != 1:
            channel = cv2.resize(channel, dsize=(int(current_w*linear_factor), int(current_h*linear_factor)),
                                   interpolation=cv2.INTER_LINEAR).astype(output_type)
        channels.append(channel)
    res_arr = np.stack(channels, axis=2)
    res_arr = np.clip(res_arr, 0, 255).astype(np.uint8)
    return np2img(res_arr)
