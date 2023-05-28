import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import random
from utils import process_images, split_dataset

focused_frames_path = 'Dataset/sharp'
defocused_frames_path = 'Dataset/defocused_blurred'

focused_frames = process_images(focused_frames_path)
defocused_frames = process_images(defocused_frames_path)

# print(len(focused_frames), len(defocused_frames))

x_train, x_test, y_train, y_test = split_dataset(defocused_frames, focused_frames)

seed = 21
random.seed = seed
np.random.seed = seed

input_shape = (128, 128, 3)
batch_size = 8
kernel_size = 3
mid_layer_dim = 256

layer_filters = [64, 128, 512]

inputs = Input(shape = input_shape, name = 'encoder_input')
x = inputs

for filters in layer_filters:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               activation='relu',
               padding='same')(x)
    
# print(x.shape)
# shape = K.init_shape(x)
shape = x.shape
x = Flatten()(x)
mid_layer = Dense(mid_layer_dim, name='mid_layer_vector')(x)

encoder = Model(inputs, mid_layer, name = 'encoder')

mid_layer_inputs = Input(shape=(mid_layer_dim), name='decoder_input')
x = Dense(shape[1] * shape[2] * shape[3])(mid_layer_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=2,
                        activation='relu',
                        padding='same')(x)
    
outputs = Conv2DTranspose(filters=3,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

decoder = Model(mid_layer_inputs, outputs, name='decoder')

autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')

# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

autoencoder.compile(loss='mse', optimizer='adam', metrics=["acc"])
lr_reducer = ReduceLROnPlateau(factor = np.sqrt(0.1),
                               cooldown = 0,
                               patience = 5,
                               verbose = 1,
                               min_lr = 0.5e-6)

callbacks = [lr_reducer]

with tf.device('/gpu:0'):
    history = autoencoder.fit(x_train,
                          y_train,
                          validation_data = (x_test, y_test),
                          epochs = 1000,
                          batch_size = batch_size,
                          callbacks = callbacks)

autoencoder.save('SavedModel/image_deblur_autoencoder_2.h5')
print("Model Saved!!!")