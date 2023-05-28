import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import random
from utils import process_images

focused_frames_path = 'Dataset/sharp'
defocused_frames_path = 'Dataset/defocused_blurred'

focused_frames = process_images(focused_frames_path)
defocused_frames = process_images(defocused_frames_path)

model = load_model('SavedModel/image_deblur_autoencoder.h5')

for i in range(3):
    
    r = random.randint(0, len(focused_frames)-1)

    x, y = defocused_frames[r], focused_frames[r]
    x_inp = x.reshape(1,128,128,3)
    with tf.device('/cpu:0'):
        result = model.predict(x_inp)
    result = result.reshape(128,128,3)

    fig = plt.figure(figsize = (12,10))
    fig.subplots_adjust(hspace = 0.1, wspace = 0.2)

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(x)

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(y)

    ax = fig.add_subplot(1, 3, 3)
    plt.imshow(result)