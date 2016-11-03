'''
Train DM Challenge classifier
GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train.py <dataset.npz> <trained_model.h5>

'''
from __future__ import print_function
import numpy as np
import sys
import cv2
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
from datetime import datetime

VGG16 = 'vgg16_weights.h5'
MODEL_PATH = 'model_{}.h5'.format(datetime.now().strftime('%Y%m%d%H%M%S'))

# training parameters
BATCH_SIZE = 10
NB_EPOCH = 5


def VGG_16(weights_path=None, input_shape=(3, 224, 224)):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    # remove last layer
    model.layers.pop()
    # model.add(Dense(1, activation='softmax'))
    model.add(Dense(1, activation='sigmoid'))
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd, loss='categorical_crossentropy')
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


# command line arguments
dataset_file = sys.argv[1]
model_file = sys.argv[2] if len(sys.argv) > 2 else MODEL_PATH

# loading dataset
print('Loading dataset')
npzfile = np.load(dataset_file)
X = npzfile['x']
Y = npzfile['y']

print('Spliting dataset')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10)

# setup model
print('Preparing model')
model = VGG_16(VGG16, X[0].shape)

# training model
print('Training model')
model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE,
          nb_epoch=NB_EPOCH,
          validation_data=(X_test, Y_test),
          shuffle=True)

# saving model
print('Saving model')
model.save(model_file)
