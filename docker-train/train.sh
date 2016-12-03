#!/usr/bin/env sh

# /modelState (writable) volume has been mounted
# /preprocessedData (writable) already mounted
# /trainingData (read-only) already mounted
# /metadata (read-only) already mounted

/usr/bin/time python train.py "/preprocessedData/dataset.h5" "/modelState/model.zip" "/scratch" 2
