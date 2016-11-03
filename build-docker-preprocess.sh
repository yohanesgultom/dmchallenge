#!/bin/sh

preprocessname="simple-preprocess"
synapseid="syn4224222"
dockerurl="docker.synapse.org"
preprocessimage="${dockerurl}/${synapseid}/${preprocessname}"

cd docker-preprocess &&
docker build -t ${preprocessimage} . &&
if [ $# -gt 0 ] && [ "$1" = --submit ]; then
    docker login ${dockerurl}
    docker push ${preprocessimage}
fi
cd ..
