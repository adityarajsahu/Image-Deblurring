import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def process_images(path):
    images = []
    for file in sorted(os.listdir(path)):
        if any(extension in file for extension in ['.jpg', '.jpeg', '.png']):
            image = load_img(path + '/' + file, target_size=(128, 128))
            image = img_to_array(image).astype('float32') / 255.0
            images.append(image)

    images = np.array(images)
    return images