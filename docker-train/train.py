'''
Train DM Challenge classifier
GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train.py <in:dataset> <out:trained_model>

'''
from __future__ import print_function
import numpy as np
import sys
import tables
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from datetime import datetime

# training parameters
BATCH_SIZE = 30
NB_SMALL = 3000
NB_EPOCH_SMALL_DATA = 30
NB_EPOCH_LARGE_DATA = 10
# CLASS_WEIGHT = {0: 0.07, 1: 1.0}
CLASS_WEIGHT = {0: 1.0, 1: 1.0}

# dataset
DATASET_BATCH_SIZE = 1000

# global consts
EXPECTED_SIZE = 224
EXPECTED_CHANNELS = 3
EXPECTED_DIM = (EXPECTED_CHANNELS, EXPECTED_SIZE, EXPECTED_SIZE)
MODEL_PATH = 'model_{}.h5'.format(datetime.now().strftime('%Y%m%d%H%M%S'))


def dataset_generator(X, Y):
    for i in range(X.nrows):
        X = dataset.data[i]
        Y = dataset.labels[i]
        yield(X, Y)


# command line arguments
dataset_file = sys.argv[1]
model_file = sys.argv[2] if len(sys.argv) > 2 else MODEL_PATH

# loading dataset
print('Loading train dataset: {}'.format(dataset_file))
datafile = tables.open_file(dataset_file, mode='r')
dataset = datafile.root
print(dataset.data[:].shape)

# determine epoch based on data size
if dataset.data[:].shape[0] <= NB_SMALL:
    NB_EPOCH = NB_EPOCH_SMALL_DATA
else:
    NB_EPOCH = NB_EPOCH_LARGE_DATA

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
# predictions = Dense(1, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# freeze base_model layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
# sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

# training model
num_rows = dataset.data.nrows
if num_rows > DATASET_BATCH_SIZE:
    # batch training
    num_iterate = num_rows / DATASET_BATCH_SIZE
    print('Training model using {} data in batch of {}'.format(num_rows, DATASET_BATCH_SIZE))
    for e in range(NB_EPOCH):
        print('Epoch {}/{}'.format(e + 1, NB_EPOCH))
        for i in range(num_iterate):
            print('Data batch {}/{}'.format(i + 1, num_iterate))
            begin = i * DATASET_BATCH_SIZE
            end = begin + DATASET_BATCH_SIZE
            X = dataset.data[begin:end]
            Y = dataset.labels[begin:end]
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10)
            model.fit(X_train, Y_train,
                      batch_size=BATCH_SIZE,
                      nb_epoch=1,
                      validation_data=(X_test, Y_test),
                      shuffle=True,
                      class_weight=CLASS_WEIGHT)
    # batch evaluate
    print('Evaluating')
    accuracies = []
    for i in range(num_iterate):
        begin = i * DATASET_BATCH_SIZE
        end = begin + DATASET_BATCH_SIZE
        X = dataset.data[begin:end]
        Y = dataset.labels[begin:end]
        s = model.evaluate(X, Y)
        accuracies.append(s[1])
    score = sum(accuracies) / float(len(accuracies))
    print('{}: {}%'.format(model.metrics_names[1], score * 100))

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
              class_weight=CLASS_WEIGHT)

    # evaluating
    print('Evaluating')
    score = model.evaluate(X_test, Y_test)
    print('{}: {}%'.format(model.metrics_names[1], score[1] * 100))

# saving model
print('Saving model')
model.save(model_file)

# close dataset
datafile.close()

print('Done.')
