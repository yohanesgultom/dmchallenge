#!/bin/bash

source config &&
docker login ${dockerurl} &&
docker push ${scoreimage}
