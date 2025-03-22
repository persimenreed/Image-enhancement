import numpy as np
from PIL import Image
from helper import np2img

def cubic_convolution(s, a=-0.5):
    if (0 <= np.abs(s)) and (np.abs(s) < 1):
        return (a+2) * np.abs(s**3) - (a+3) * np.abs(s**2) + 1
    elif (1 <= np.abs(s)) and (np.abs(s) < 2):
        return a * np.abs(s)**3 - 5*a * np.abs(s)**2 + 8*a * np.abs(s) - 4*a
    else:
        return 0
        
def imresize_bicubic(im, width, height, img_obj=True):
    """
    This function resizes the input Image object im to the size of (width x height) 
    pixels based on the bicubic interpolation and returns the resized Image object.
    """
    # Convert the input Image object to numpy array object

    im = np.array(im).astype(float)
    if (im.shape[0] == height) and (im.shape[1] == width):
        im_res = im.copy()
    else:
        h, w = im.shape[0], im.shape[1]
        # Padding the image to ensure all pixels fit in the middle of a 4x4 kernel
        im_padded = np.zeros((h+4, w+4, 3))
        im_padded[2:h+2, 2:w+2] = im
        im_padded[0:2, 2:w+2] = im[0, :]
        im_padded[h+2:h+4, 2:w+2] = im[h-1, :]
        im_padded[2:h+2, 0:2] = im[:, 0:1]
        im_padded[2:h+2, w+2:w+4] = im[:, w-1:w]

        im_padded[0:2, 0:2] = im[0, 0]
        im_padded[0:2, w+2:w+4] = im[0, w-1]
        im_padded[h+2:h+4, 0:2] = im[h-1, 0]
        im_padded[h+2:h+4, w+2:w+4] = im[h-1, w-1]

        im_res = np.zeros([height, width, im.shape[2]])


        # Convolution
        for hr in range(height):
            for wr in range(width):
                for c in range(im.shape[2]):
                    x, y = (wr + 0.5) * (w / width) - 0.5 + 2, (hr + 0.5) * (h / height) - 0.5 + 2
                    
                    x_floor = int(np.floor(x))
                    y_floor = int(np.floor(y))

                    dx = x - x_floor
                    dy = y - y_floor

                    matrix_x = np.array([cubic_convolution(dx+1), cubic_convolution(dx), cubic_convolution(dx-1), cubic_convolution(dx-2)])
                    matrix_y = np.array([cubic_convolution(dy+1), cubic_convolution(dy), cubic_convolution(dy-1), cubic_convolution(dy-2)])
                    matrix_a = im_padded[y_floor-1:y_floor+3, x_floor-1:x_floor+3, c]
                    

                    im_res[hr, wr, c] = np.dot(np.dot(matrix_x, matrix_a.T), matrix_y)
        im_res = np.maximum(0, np.minimum(255, im_res))


    if img_obj:
        im_res = np2img(im_res)
    
    return im_res








