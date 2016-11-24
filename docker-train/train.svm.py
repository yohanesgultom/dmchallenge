import numpy as np
import sys
from tables import open_file
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

# training parameters
BATCH_SIZE = 5


def dataset_generator(dataset, batch_size):
    while True:
        for i in range(dataset.data.nrows):
            X = dataset.data[i: i + batch_size]
            Y = dataset.labels[i: i + batch_size]
            yield(X, Y)

dataset_file = sys.argv[1]

print('Loading train dataset: {}'.format(dataset_file))
datafile = open_file(dataset_file, mode='r')
dataset = datafile.root
print(dataset.data[:].shape)

print('Loading feature extractor')
model = VGG16(weights='imagenet', include_top=False)

print('Extracting features')

for i in range(dataset.data.nrows):
    X = dataset.data[i: i + BATCH_SIZE]
    Y = dataset.labels[i: i + BATCH_SIZE]
    batch_prediction = model.predict(X)
    break

# TODO svm

datafile.close()
print('Done.')
