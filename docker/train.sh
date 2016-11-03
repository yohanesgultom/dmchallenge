#!/bin/sh
modelname="simple-cnn-vgg16"

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train.py "/preprocessedData/dataset.npz" "/modelState/${modelname}.h5"
