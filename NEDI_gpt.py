import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import sys
from PIL import Image

"""
Author:
    hu.leying@columbia.edu

Usage:
    EDI_predict(img, m, s)
      - img is the input PIL image (color or grayscale)
      - m is the sampling window size (ideal m>=4)
      - s is the scaling factor (s > 0, e.g. s=2 to upscale by 2, s=0.5 to downscale by 2)
      
Direct calls:
    EDI_upscale(img, m)   -> upscales a grayscale image by a factor of 2
    EDI_downscale(img)    -> downscales a grayscale image by a factor of 2
"""

def EDI_downscale(img):
    # img is a 2D array (grayscale)
    h, w = img.shape  # h: height, w: width
    imgo2 = np.zeros((h//2, w//2))
    for i in range(h//2):
        for j in range(w//2):
            imgo2[i, j] = int(img[2*i, 2*j])
    return imgo2.astype(img.dtype)

def EDI_upscale(img, m):
    # img is a 2D array (grayscale); m should be a power of 2 (if odd, increment by 1)
    if m % 2 != 0:
        m += 1
    h, w = img.shape  # h: height, w: width
    imgo = np.zeros((h*2, w*2))
    
    # Place low-res pixels into even indices.
    for i in range(h):
        for j in range(w):
            imgo[2*i, 2*j] = img[i, j]
    
    y = np.zeros((m**2, 1))
    C = np.zeros((m**2, 4))
    
    # Reconstruct pixels at positions (2*i+1, 2*j+1)
    for i in range(math.floor(m/2), h - math.floor(m/2)):
        for j in range(math.floor(m/2), w - math.floor(m/2)):
            tmp = 0
            for ii in range(i - math.floor(m/2), i + math.floor(m/2)):
                for jj in range(j - math.floor(m/2), j + math.floor(m/2)):
                    y[tmp, 0] = imgo[2*ii, 2*jj]
                    C[tmp, 0] = imgo[2*ii-2, 2*jj-2] if (2*ii-2 >= 0 and 2*jj-2 >= 0) else imgo[2*ii, 2*jj]
                    C[tmp, 1] = imgo[2*ii+2, 2*jj-2] if (2*ii+2 < imgo.shape[0] and 2*jj-2 >= 0) else imgo[2*ii, 2*jj]
                    C[tmp, 2] = imgo[2*ii+2, 2*jj+2] if (2*ii+2 < imgo.shape[0] and 2*jj+2 < imgo.shape[1]) else imgo[2*ii, 2*jj]
                    C[tmp, 3] = imgo[2*ii-2, 2*jj+2] if (2*ii-2 >= 0 and 2*jj+2 < imgo.shape[1]) else imgo[2*ii, 2*jj]
                    tmp += 1
            a = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(C), C)), np.transpose(C)), y)
            vec = np.array([
                imgo[2*i,   2*j],
                imgo[2*i+2, 2*j]   if (2*i+2 < imgo.shape[0]) else imgo[2*i, 2*j],
                imgo[2*i+2, 2*j+2] if (2*i+2 < imgo.shape[0] and 2*j+2 < imgo.shape[1]) else imgo[2*i, 2*j],
                imgo[2*i,   2*j+2] if (2*j+2 < imgo.shape[1]) else imgo[2*i, 2*j]
            ])
            imgo[2*i+1, 2*j+1] = np.dot(vec, a[:, 0])
    
    # Reconstruct pixels at positions (2*i+1, 2*j) and (2*i, 2*j+1)
    for i in range(math.floor(m/2), h - math.floor(m/2)):
        for j in range(math.floor(m/2), w - math.floor(m/2)):
            tmp = 0
            for ii in range(i - math.floor(m/2), i + math.floor(m/2)):
                for jj in range(j - math.floor(m/2), j + math.floor(m/2)):
                    y[tmp, 0] = imgo[2*ii+1, 2*jj-1]
                    C[tmp, 0] = imgo[2*ii-1, 2*jj-1] if (2*ii-1 >= 0 and 2*jj-1 >= 0) else imgo[2*ii+1, 2*jj-1]
                    C[tmp, 1] = imgo[2*ii+1, 2*jj-3] if (2*jj-3 >= 0) else imgo[2*ii+1, 2*jj-1]
                    C[tmp, 2] = imgo[2*ii+3, 2*jj-1] if (2*ii+3 < imgo.shape[0]) else imgo[2*ii+1, 2*jj-1]
                    C[tmp, 3] = imgo[2*ii+1, 2*jj+1] if (2*jj+1 < imgo.shape[1]) else imgo[2*ii+1, 2*jj-1]
                    tmp += 1
            a = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(C), C)), np.transpose(C)), y)
            imgo[2*i+1, 2*j] = np.matmul([imgo[2*i, 2*j],
                                           imgo[2*i+1, 2*j-1] if (2*j-1 >= 0) else imgo[2*i, 2*j],
                                           imgo[2*i+2, 2*j]   if (2*i+2 < imgo.shape[0]) else imgo[2*i, 2*j],
                                           imgo[2*i+1, 2*j+1] if (2*j+1 < imgo.shape[1]) else imgo[2*i, 2*j]], a)
            imgo[2*i, 2*j+1] = np.matmul([imgo[2*i-1, 2*j+1] if (2*i-1 >= 0) else imgo[2*i, 2*j],
                                           imgo[2*i, 2*j],
                                           imgo[2*i+1, 2*j+1] if (2*i+1 < imgo.shape[0] and 2*j+1 < imgo.shape[1]) else imgo[2*i, 2*j],
                                           imgo[2*i, 2*j+2] if (2*j+2 < imgo.shape[1]) else imgo[2*i, 2*j]], a)
    
    # Fill remaining (boundary) zero entries using bilinear interpolation.
    np.clip(imgo, 0, 255.0, out=imgo)
    imgo_bilinear = cv2.resize(img, dsize=(w*2, h*2), interpolation=cv2.INTER_LINEAR)
    imgo[imgo == 0] = imgo_bilinear[imgo == 0]
    
    return imgo.astype(img.dtype)

def EDI_predict_channel(arr, m=4, s=4):
    """
    Applies EDI prediction to a single-channel (grayscale) image (NumPy array).
    
    Parameters:
      arr : 2D NumPy array representing a grayscale image.
      m   : Sampling window size.
      s   : Scaling factor.
      
    Returns:
      A 2D NumPy array of the rescaled image.
    """
    h, w = arr.shape  # h: height, w: width
    output_type = arr.dtype

    if s <= 0:
        sys.exit("Error input: Please input s > 0!")
    elif s == 1:
        return arr
    elif s < 1:
        n = math.floor(math.log(1/s, 2))
        linear_factor = 1/s / (2**n)
        if linear_factor != 1:
            arr = cv2.resize(arr, dsize=(int(w/linear_factor), int(h/linear_factor)), interpolation=cv2.INTER_LINEAR).astype(output_type)
        for i in range(n):
            arr = EDI_downscale(arr)
        return arr
    elif s < 2:
        arr_res = cv2.resize(arr, dsize=(int(w*s), int(h*s)), interpolation=cv2.INTER_LINEAR).astype(output_type)
        return arr_res
    else:
        n = math.floor(math.log(s, 2))
        for i in range(n):
            arr = EDI_upscale(arr, m)
        current_h, current_w = arr.shape
        linear_factor = s / (2**n)
        if linear_factor != 1:
            arr = cv2.resize(arr, dsize=(int(current_w*linear_factor), int(current_h*linear_factor)), interpolation=cv2.INTER_LINEAR).astype(output_type)
        return arr

def EDI_predict(img, m=4, s=4):
    """
    Main prediction function.
    Accepts any input PIL Image (color or grayscale), processes each channel individually if needed,
    and returns a rescaled PIL Image.
    
    Parameters:
      img : Input PIL Image.
      m   : Sampling window size (ideal m >= 4).
      s   : Scaling factor (s > 0).
    
    Returns:
      A rescaled PIL Image. Color images are processed per-channel.
    """
    # Convert image to a NumPy array.
    arr = np.array(img).astype(np.float64)
    
    # If the image is color (3D with 3 channels), process each channel individually.
    if arr.ndim == 3 and arr.shape[2] == 3:
        channels = []
        for c in range(3):
            ch = EDI_predict_channel(arr[:, :, c], m, s)
            channels.append(ch)
        # Stack channels along the third axis.
        res_arr = np.stack(channels, axis=2)
        # Ensure values are in 0-255 and convert to uint8.
        res_arr = np.clip(res_arr, 0, 255).astype(np.uint8)
        return Image.fromarray(res_arr, mode='RGB')
    else:
        # For grayscale images:
        res_arr = EDI_predict_channel(arr, m, s)
        res_arr = np.clip(res_arr, 0, 255).astype(np.uint8)
        return Image.fromarray(res_arr, mode='L')
