'''
Train DM Challenge classifier
GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train.py <in:dataset> <out:trained_model>

'''
from __future__ import print_function
import numpy as np
import sys
import tables
import keras.backend as K
import json
import os
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from datetime import datetime

# training parameters
BATCH_SIZE = 40
NB_SMALL = 3000
NB_EPOCH_SMALL_DATA = 12
NB_EPOCH_LARGE_DATA = 10

# dataset
DATASET_BATCH_SIZE = 1000

# global consts
EXPECTED_SIZE = 224
EXPECTED_CHANNELS = 3
EXPECTED_DIM = (EXPECTED_CHANNELS, EXPECTED_SIZE, EXPECTED_SIZE)
MODEL_PATH = 'model_{}.zip'.format(datetime.now().strftime('%Y%m%d%H%M%S'))


def dataset_generator(dataset, batch_size):
    while True:
        for i in range(dataset.data.nrows):
            X = dataset.data[i: i + batch_size]
            Y = dataset.labels[i: i + batch_size]
            yield(X, Y)


def confusion(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos) / (K.sum(y_pos) + K.epsilon())
    tn = K.sum(y_neg * y_pred_neg) / (K.sum(y_neg) + K.epsilon())
    return {'true_pos': tp, 'true_neg': tn}

# command line arguments
dataset_file = sys.argv[1]
model_file = sys.argv[2] if len(sys.argv) > 2 else MODEL_PATH
verbosity = int(sys.argv[3]) if len(sys.argv) > 3 else 1

# loading dataset
print('Loading train dataset: {}'.format(dataset_file))
datafile = tables.open_file(dataset_file, mode='r')
dataset = datafile.root
print(dataset.data[:].shape)

# determine training params based on data size
if dataset.data[:].shape[0] <= NB_SMALL:
    NB_EPOCH = NB_EPOCH_SMALL_DATA
else:
    NB_EPOCH = NB_EPOCH_LARGE_DATA

# set class_weight dynamically
ratio = dataset.ratio[0]
class_weight = {0: ratio[0], 1: ratio[1]}

print('BATCH_SIZE: {}'.format(BATCH_SIZE))
print('NB_EPOCH: {}'.format(NB_EPOCH))
print('class_weight: {}'.format(class_weight))

# setup model
print('Preparing model')
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=EXPECTED_DIM))
x = base_model.output
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid', init='uniform')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# freeze base_model layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy', confusion])

# training model
num_rows = dataset.data.nrows
if num_rows > DATASET_BATCH_SIZE:
    # batch training
    model.fit_generator(
        dataset_generator(dataset, BATCH_SIZE),
        samples_per_epoch=num_rows,
        nb_epoch=NB_EPOCH,
        class_weight=class_weight,
        verbose=verbosity
    )

else:
    # one-go training
    X = dataset.data[:]
    Y = dataset.labels[:]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10)
    model.fit(X_train, Y_train,
              batch_size=BATCH_SIZE,
              nb_epoch=NB_EPOCH,
              validation_data=(X_test, Y_test),
              shuffle=True,
              verbose=verbosity,
              class_weight=class_weight)

# saving model weights and architecture only
# to save space
print('Saving model')
model_name = os.path.basename(model_file)
model_path = os.path.splitext(model_file)[0]
weights_file = model_path + '.weights.h5'
arch_file = model_path + '.arch.json'
model.save_weights(weights_file)
with open(arch_file, 'w') as outfile:
    outfile.write(model.to_json())

# batch evaluate
print('Evaluating')
score = model.evaluate_generator(dataset_generator(dataset, BATCH_SIZE), num_rows)
for i in range(1, len(model.metrics_names)):
    print('{}: {}%'.format(model.metrics_names[i], score[i] * 100))

# close dataset
datafile.close()

print('Done.')
