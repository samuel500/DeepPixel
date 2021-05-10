



import os
import sys

import tensorflow as tf

import numpy as np
from matplotlib.pyplot import imread

import imageio

from matplotlib import pyplot as plt

import imageio

from tqdm import tqdm
import cv2

import scipy.ndimage as nd

from image_utils import load_image, show

from models_utils import *

from squeezenet import SqueezeNet
from utils import weight_to_weight, get_ckpt_weights




m = SqueezeNet()
m.summary()

all_n = []
    
print(os.getcwd())

all_n = get_ckpt_weights('./squeezenet_weights/squeezenet.ckpt')

all_weights = []
for w in m.model.weights:
    all_weights.append(all_n[weight_to_weight[w.name]])

m.set_weights(all_weights)
m.trainable = False


names = ['fire_layer_4']

# print([l.name for l in m.model.layers])

layers = [m.model.get_layer(name).output for name in names]

if len(layers) == 1:
    layers = layers[0]

# Create the feature extraction model
dream_model = tf.keras.Model(inputs=m.model.input, outputs=layers)


dream_model.preprocess_image = m.preprocess_input
dream_model.deprocess_image = deprocess
dream_model.clip_image = lambda img: tf.clip_by_value(img, -1., 1.)