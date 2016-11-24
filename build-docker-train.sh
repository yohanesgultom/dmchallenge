#!/bin/bash

source config
cd docker-train
cp $HOME/.keras/models/vgg16_weights_th_dim_ordering_th_kernels_notop.h5 .
docker build -t ${trainimage} .
cd ..
