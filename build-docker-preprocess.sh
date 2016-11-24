#!/bin/bash

source config
cd docker-preprocess
docker build -t ${preprocessimage} .
cd ..
