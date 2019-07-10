# CNN model for categorizing videos
# Ashley Fletcher

from __future__ import absolute_import, division, print_function, unicode_literals

import pickle
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD

tf.enable_eager_execution()

# first convolutional layers
# should each output shape 55x55x96
frames = [''] *20
inputs = [''] *20
print(frames)

for i in range(20):
    inputs[i] = tf.keras.Input(shape=(227,227,3,))
    frames[i] = tf.keras.layers.Conv2D(96, 11, input_shape=(227, 227, 3), strides=4, activation='relu' )(inputs[i])

# add the 20 frames into each other
# should output shape 55x55x96 
added = tf.keras.layers.Add()( frames )


# normalization
norm1 = layers.BatchNormalization()( added )
# first pooling layer
pool1 = layers.MaxPool2D( pool_size=3, strides=2 )( norm1 )


# first 2D convolutional
conv2 = layers.Conv2D(256, 5, strides=1 )( pool1 )
# normalization
norm2 = layers.BatchNormalization()(conv2)
# second pooling
pool2 = layers.MaxPool2D( pool_size=3, strides=2 )( norm2 )


# third convolutional layer
conv3 = layers.Conv2D(384, 3, strides=1 )( pool2 )
# fourth convolutional
conv4 = layers.Conv2D(384, 3, strides=1 )( conv3 )
# fifth convolutional
conv5 = layers.Conv2D(256, 3, strides=1 )( conv4 )
# third pooling
pool3 = layers.MaxPool2D( pool_size=3, strides=2 )( conv5 )

# 
flat = layers.Flatten()( pool3 ) # set to pool3 when uncomment previous lines

# three fully-connected layers
fc1 = layers.Dense( units=512 )( flat ) # i reduced the number of units for testing on my laptop
fc2 = layers.Dense( units=512 )( fc1 )
fc3 = layers.Dense( units=101 )( fc2 )

# softmax
soft = layers.Activation( 'softmax' )( fc3 )

model = tf.keras.Model( inputs=inputs, outputs=soft )

opt = SGD(lr=0.01)

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# time = (datetime.now()).strftime('%Y-%m-%d-%H:%M')
filename = 'model-' # + time
output_file = open(filename, 'wb')
pickle.dump(model, output_file)
output_file.close()
