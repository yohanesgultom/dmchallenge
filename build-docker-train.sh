#!/bin/bash

source config &&
cd docker-train &&
docker build -t ${trainimage} . &&
cd ..
