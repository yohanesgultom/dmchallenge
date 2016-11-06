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
echo $(nvcc --version)

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train.py "/preprocessedData/dataset.npz" "/modelState/${modelname}.h5"
