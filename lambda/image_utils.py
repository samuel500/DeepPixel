
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import imread

#from scipy.misc import imresize

from cv2 import resize as imresize

def load_image(filename):
    """Load and resize an image from disk."""
    img = imread(filename)

    if len(img.shape) > 2 and img.shape[2] == 4:
        img = img[:,:,:3] # remove alpha channel

    return img

# Display an image
def show(img):
    plt.figure(figsize=(12,12))
    plt.grid(False)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
