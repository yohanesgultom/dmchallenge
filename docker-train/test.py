import sys
import dicom
import csv
from keras.models import load_model

weights_file = sys.argv[1]
images_dir = sys.argv[2]
meta_file = sys.argv[3]

weights = load_weights(weights_file)

# TODO
# score = model.evaluate(X, Y)
# print('{}: {}%'.format(model.metrics_names[1], score[1] * 100))
