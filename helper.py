import cv2
from PIL import Image
import numpy as np

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

def downsampling(img, factor):
    large_img = cv2.imread(img)
    large_img = cv2.cvtColor(large_img, cv2.COLOR_BGR2RGB)
    small_img = cv2.resize(large_img,
                           (0,0),
                           fx = factor,
                           fy = factor,
                           interpolation = cv2.INTER_NEAREST)
    
    small_img = np2img(small_img)

    return small_img