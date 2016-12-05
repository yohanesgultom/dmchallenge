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
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from datetime import datetime

# training parameters
BIG_BATCH_SIZE = 1000  # batch size for data > NB_DATA_LIMIT
SMALL_BATCH_SIZE = 40  # for data <= NB_DATA_LIMIT
NB_EPOCH = 30
NB_DATA_LIMIT = 3000

# global consts
EXPECTED_SIZE = 224
EXPECTED_CHANNELS = 3
EXPECTED_CLASS = 1
EXPECTED_DIM = (EXPECTED_CHANNELS, EXPECTED_SIZE, EXPECTED_SIZE)
MODEL_PATH = 'model.zip'
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
FEATURES_FILE = 'features.h5'
FEATURES_DIM = (512, 7, 7)


def dataset_generator(dataset, batch_size):
    while True:
        i = 0
        while i < dataset.data.nrows:
            end = i + batch_size
            X = dataset.data[i:end]
            Y = dataset.labels[i:end]
            i = end
            yield(X, Y)


def h5_generator(data, batch_size):
    while True:
        i = 0
        while i < data.nrows:
            end = i + batch_size
            chunk = data[i:end]
            i = end
            yield(chunk)


def confusion(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos) / (K.sum(y_pos) + K.epsilon())
    tn = K.sum(y_neg * y_pred_neg) / (K.sum(y_neg) + K.epsilon())
    return {'true_pos': tp, 'true_neg': tn}


if __name__ == '__main__':
    # command line arguments
    dataset_file = sys.argv[1]
    model_file = sys.argv[2] if len(sys.argv) > 2 else MODEL_PATH
    scratch_dir = sys.argv[3] if len(sys.argv) > 2 else CUR_DIR
    verbosity = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    extract_verbosity = 1 if verbosity == 1 else 0

    # loading dataset
    print('Loading train dataset: {}'.format(dataset_file))
    datafile = tables.open_file(dataset_file, mode='r')
    dataset = datafile.root
    print(dataset.data[:].shape)

    # set class_weight dynamically
    ratio = dataset.ratio[0]
    class_weight = {0: ratio[0], 1: ratio[1]}

    print('NB_EPOCH: {}'.format(NB_EPOCH))
    print('class_weight: {}'.format(class_weight))

    # feature extractor
    extractor = VGG16(weights='imagenet', include_top=False)

    # this is the model we will train
    model = Sequential()
    model.add(Flatten(input_shape=FEATURES_DIM))
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', init='uniform'))

    # compile the model (should be done *after* setting layers to non-trainable)
    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy', confusion])

    # early stopping
    # early_stopping_acc = EarlyStopping(monitor='acc', patience=3)

    # feature extraction and model training
    num_rows = dataset.data.nrows
    if num_rows > NB_DATA_LIMIT:
        # batch feature extraction
        print('BIG_BATCH_SIZE: {}'.format(BIG_BATCH_SIZE))
        print('Batch feature extraction')
        features_file_path = os.path.join(scratch_dir, FEATURES_FILE)
        features_file = tables.open_file(features_file_path, mode='w')
        features_data = features_file.create_earray(features_file.root, 'data', tables.Float32Atom(shape=FEATURES_DIM), (0,), 'dream')
        features_labels = features_file.create_earray(features_file.root, 'labels', tables.UInt8Atom(shape=(EXPECTED_CLASS)), (0,), 'dream')
        i = 0
        while i < dataset.data.nrows:
            end = i + BIG_BATCH_SIZE
            data_chunk = dataset.data[i:end]
            label_chunk = dataset.labels[i:end]
            i = end
            features_data.append(extractor.predict(data_chunk, verbose=extract_verbosity))
            features_labels.append(label_chunk)

        assert features_file.root.data.nrows == num_rows
        assert features_file.root.labels.nrows == dataset.labels.nrows

        # batch training
        print('Batch training')
        model.fit_generator(
            dataset_generator(features_file.root, BIG_BATCH_SIZE),
            samples_per_epoch=num_rows,
            nb_epoch=NB_EPOCH,
            class_weight=class_weight,
            verbose=verbosity
        )

        predictions = model.predict_generator(h5_generator(features_file.root.data, BIG_BATCH_SIZE), val_samples=num_rows)
        Y = features_file.root.labels[:]

        # close feature file
        features_file.close()
        os.remove(features_file_path)

    else:
        # one-go feature extraction and model training
        print('SMALL_BATCH_SIZE: {}'.format(SMALL_BATCH_SIZE))
        print('Feature extraction')
        X = extractor.predict(dataset.data[:], verbose=1)
        Y = dataset.labels[:]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10)
        print('Training')
        model.fit(X_train, Y_train,
                  batch_size=SMALL_BATCH_SIZE,
                  nb_epoch=NB_EPOCH,
                  validation_data=(X_test, Y_test),
                  shuffle=True,
                  verbose=verbosity,
                  class_weight=class_weight)
        predictions = model.predict(X)

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

    # evaluate on training data
    print('Evaluate on training data')
    predictions = np.round(predictions).astype('int')
    acc = accuracy_score(Y, predictions)
    roc_auc = roc_auc_score(Y, predictions)
    print('Accuracy: {}'.format(acc))
    print('ROC AUC: {}'.format(roc_auc))
    print('Confusion matrix: ')
    cm = confusion_matrix(Y, predictions)
    print(cm)
    print('True Positive: {}'.format(cm[1][1] * 1.0 / (cm[1][1] + cm[1][0])))
    print('True Negative: {}'.format(cm[0][0] * 1.0 / (cm[0][0] + cm[0][1])))

    # close dataset
    datafile.close()

    print('Done.')
