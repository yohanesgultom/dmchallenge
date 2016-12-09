import sys
import tables
import numpy as np
from keras.models import model_from_json
from keras.applications.vgg16 import VGG16
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# command args
dataset_file = sys.argv[1]
arch_file = sys.argv[2]
weights_file = sys.argv[3]

# loading dataset
print('Loading evaluation data: {}'.format(dataset_file))
datafile = tables.open_file(dataset_file, mode='r')
dataset = datafile.root
print(dataset.data[:].shape)

# feature extractor
print('Loading extractor and model')
extractor = VGG16(weights='imagenet', include_top=False)

# load model
with open(arch_file) as f:
    arch_json = f.read()
    model = model_from_json(arch_json)
model.load_weights(weights_file)

# extract features
print('Extracting features')
X = extractor.predict(datafile.root.data[:], verbose=1)
Y = datafile.root.labels[:]

print('Predicting')
predictions = np.round(model.predict_on_batch(X)).astype('int')
acc = accuracy_score(Y, predictions)
roc_auc = roc_auc_score(Y, predictions)
print('Accuracy: {}'.format(acc))
print('ROC AUC: {}'.format(roc_auc))
print('Confusion matrix: ')
cm = confusion_matrix(Y, predictions)
print(cm)
print('True Positive: {}'.format(cm[1][1] * 1.0 / (cm[1][1] + cm[1][0])))
print('True Negative: {}'.format(cm[0][0] * 1.0 / (cm[0][0] + cm[0][1])))
