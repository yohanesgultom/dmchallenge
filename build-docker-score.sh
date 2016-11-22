#!/bin/bash

source config
cd docker-score
cp . ../docker-preprocess/preprocess.py
docker build -t ${scoreimage} .
cd ..
