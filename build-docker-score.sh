#!/bin/bash

source config
cd docker-score
cp ../docker-preprocess/preprocess.py .
cp ../docker-train/keras.json .
cp ../docker-train/.theanorc .
cp ../docker-train/vgg16_weights_th_dim_ordering_th_kernels_notop.h5 .
docker build -t ${scoreimage} .
cd ..
