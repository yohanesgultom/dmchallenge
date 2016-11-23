#!/bin/bash

source config
cd docker-score
cp ../docker-preprocess/preprocess.py .
cp ../docker-train/keras.json .
cp ../docker-train/.theanorc .
docker build -t ${scoreimage} .
cd ..
