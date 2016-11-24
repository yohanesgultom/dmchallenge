import tables
import sys
from numpy import transpose
from cv2 import imshow, waitKey

dataset_file = sys.argv[1]
datafile = tables.open_file(dataset_file, mode='r')
dataset = datafile.root

for data in dataset.data:
    data = transpose(data, (1, 2, 0))
    print(data.shape)
    imshow('image', data)
    waitKey(0)

datafile.close()
