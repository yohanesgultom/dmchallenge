#!/usr/bin/env sh

# /modelState (writable) volume has been mounted
# /preprocessedData (writable) already mounted
# /trainingData (read-only) already mounted
# /metadata (read-only) already mounted

# vars
modelname="simple-cnn-vgg16"

# display specs
echo "$USER"
echo $(free -m)
echo $(nvidia-smi)

python train.py "/preprocessedData/dataset.npz" "/modelState/${modelname}.h5"
