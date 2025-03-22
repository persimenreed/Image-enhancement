import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import sys
from PIL import Image

"""
Template used as baseline for this implementation: https://github.com/Kirstihly/Edge-Directed_Interpolation
"""

def np2img(im, norm=False, rgb_mode=False):
    """
    This function converts the input numpy object im to Image object and returns
    the converted object. If norm == True, then the input is normalised to [0,1]
    using im <- (im - im.min()) / (im.max() - im.min()).
    """
    if norm:
        if ((im.max() - im.min()) != 0.0):
            im = (im - im.min()) / (im.max() - im.min())

    if ((im.min() >= 0.0) and (im.max() <= 1.0)):
        im = im * 255.0

    if rgb_mode and im.ndim == 2:
        im = im[...,np.newaxis].repeat(3, axis=2)
        
    if im.ndim == 2:
        im = Image.fromarray(im.astype(np.uint8), mode='L')
    elif (im.ndim == 3) and (im.shape[2] == 3):
        im = Image.fromarray(im.astype(np.uint8), mode='RGB')

    return im

def NEDI_upscale(img, m):

    # initializing the image to be predicted
    h, w = img.shape
    imgo = np.zeros((h*2, w*2))
    
    # Place low-resolution pixels
    for i in range(h):
        for j in range(w):
            imgo[2*i, 2*j] = img[i, j]
    
    y = np.zeros((m**2, 1)) # This is the pixels in the window
    C = np.zeros((m**2, 4)) # C is the list of interpolation neighbour for each pixel in the window
    
    # Reconstruct the points with the form of (2*i+1,2*j+1)
    for i in range(math.floor(m/2), h - math.floor(m/2)):
        for j in range(math.floor(m/2), w - math.floor(m/2)):
            tmp = 0
            for ii in range(i - math.floor(m/2), i + math.floor(m/2)):
                for jj in range(j - math.floor(m/2), j + math.floor(m/2)):
                    y[tmp, 0] = imgo[2*ii, 2*jj]
                    C[tmp, 0] = imgo[2*ii-2, 2*jj-2]
                    C[tmp, 1] = imgo[2*ii+2, 2*jj-2]
                    C[tmp, 2] = imgo[2*ii+2, 2*jj+2]
                    C[tmp, 3] = imgo[2*ii-2, 2*jj+2]
                    tmp += 1

            # calculating weights
            # a = (C^T * C)^(-1) * (C^T * y) = (C^T * C) \ (C^T * y)
            a = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(C),C)), np.transpose(C)), y)
            imgo[2*i+1][2*j+1] = np.matmul([imgo[2*i][2*j], imgo[2*i+2][2*j], imgo[2*i+2][2*j+2], imgo[2*i][2*j+2]], a)
    
    # Reconstructed the points with the forms of (2*i+1,2*j) and (2*i,2*j+1)
    for i in range(math.floor(m/2), h - math.floor(m/2)):
        for j in range(math.floor(m/2), w - math.floor(m/2)):
            tmp = 0
            for ii in range(i - math.floor(m/2), i + math.floor(m/2)):
                for jj in range(j - math.floor(m/2), j + math.floor(m/2)):
                    y[tmp, 0] = imgo[2*ii+1, 2*jj-1]
                    C[tmp, 0] = imgo[2*ii-1, 2*jj-1]
                    C[tmp, 1] = imgo[2*ii+1, 2*jj-3]
                    C[tmp, 2] = imgo[2*ii+3, 2*jj-1]
                    C[tmp, 3] = imgo[2*ii+1, 2*jj+1]
                    tmp += 1

            # calculating weights
            # a = (C^T * C)^(-1) * (C^T * y) = (C^T * C) \ (C^T * y)
            a = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(C),C)), np.transpose(C)), y)
            imgo[2*i+1][2*j] = np.matmul([imgo[2*i][2*j], imgo[2*i+1][2*j-1], imgo[2*i+2][2*j], imgo[2*i+1][2*j+1]], a)
            imgo[2*i][2*j+1] = np.matmul([imgo[2*i-1][2*j+1], imgo[2*i][2*j], imgo[2*i+1][2*j+1], imgo[2*i][2*j+2]], a)

    # the rest of the pixels are filled with bilinear interpolation
    np.clip(imgo, 0, 255.0, out=imgo)
    imgo_bilinear = cv2.resize(img, dsize=(w*2, h*2), interpolation=cv2.INTER_LINEAR)
    imgo[imgo == 0] = imgo_bilinear[imgo == 0]
    
    return imgo.astype(img.dtype)

def NEDI_predict(img, m=4, s=4):
    arr = np.array(img).astype(np.float64)
    output_type = arr.dtype

    channels = []
    for c in range(3):
        channel = arr[:, :, c]
        h, w = channel.shape
        n = math.floor(math.log(s, 2))
        for _ in range(n):
            channel = NEDI_upscale(channel, m)
        current_h, current_w = channel.shape
        linear_factor = s / (2**n)
        if linear_factor != 1:
            channel = cv2.resize(channel, dsize=(int(current_w*linear_factor), int(current_h*linear_factor)),
                                   interpolation=cv2.INTER_LINEAR).astype(output_type)
        channels.append(channel)
    res_arr = np.stack(channels, axis=2)
    res_arr = np.clip(res_arr, 0, 255).astype(np.uint8)
    return np2img(res_arr)
