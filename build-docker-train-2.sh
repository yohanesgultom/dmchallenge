#!/bin/bash

source config
cd docker-train-2
cp $HOME/.keras/models/vgg16_weights_th_dim_ordering_th_kernels_notop.h5 .
docker build -t ${trainimage2} .
cd ..
