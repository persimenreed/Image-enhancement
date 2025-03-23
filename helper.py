from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt

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

def downsampling(image_path, factor):
    img = Image.open(image_path)
    new_size = (int(img.width * factor), int(img.height * factor))
    return img.resize(new_size, Image.BICUBIC)

# https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

def SSIM(original, compressed):
    if original.ndim == 3:
        original = np.dot(original[..., :3], [0.2989, 0.5870, 0.1140])
    if compressed.ndim == 3:
        compressed = np.dot(compressed[..., :3], [0.2989, 0.5870, 0.1140])
    return ssim(original, compressed, data_range=compressed.max() - compressed.min())