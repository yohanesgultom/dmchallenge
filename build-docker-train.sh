#!/bin/sh

trainname="simple-cnn-vgg16"
synapseid="syn4224222"
dockerurl="docker.synapse.org"
trainimage="${dockerurl}/${synapseid}/${trainname}"

cd docker-train &&
docker build -t ${trainimage} . &&
if [ $# -gt 0 ] && [ "$1" = --submit ]; then
    docker login ${dockerurl}
    docker push ${trainimage}
fi
cd ..
