import sys
import tables
from keras.models import load_model

model_file = sys.argv[1]
test_file = sys.argv[2]

model = load_model(model_file)
datafile = tables.open_file(test_file, mode='r')
X = datafile.root.data[:]
Y = datafile.root.labels[:]
score = model.evaluate(X, Y)
print('{}: {}%'.format(model.metrics_names[1], score[1] * 100))
datafile.close()
