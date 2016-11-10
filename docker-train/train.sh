#!/usr/bin/env sh

# /modelState (writable) volume has been mounted
# /preprocessedData (writable) already mounted
# /trainingData (read-only) already mounted
# /metadata (read-only) already mounted

# vars
modelname="simple-cnn-vgg16"

# display specs
echo $(free -m)
printf "\n\n"
echo $(nvidia-smi)
printf "\n\n"

python train.py "/preprocessedData/dataset.h5" "/modelState/${modelname}.h5"
