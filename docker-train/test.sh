#!/usr/bin/env sh

# /modelState (writable) volume has been mounted
# /preprocessedData (writable) already mounted
# /trainingData (read-only) already mounted
# /metadata (read-only) already mounted

source config

/usr/bin/time python test.py "/modelState/${modelname}.h5" "/metadata" "/testData"
