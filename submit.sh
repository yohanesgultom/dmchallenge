#!/bin/sh

modelname="simple-cnn-vgg16"
synapseid="syn4224222"
dockerurl="docker.synapse.org"
image="${dockerurl}/${synapseid}/${modelname}"


cd docker &&
docker build -t ${image} . &&
if [ $# -gt 0 ] && [ "$1" = --submit ]; then
    # docker login ${dockerurl}
    # docker push ${image}
fi
cd ..
