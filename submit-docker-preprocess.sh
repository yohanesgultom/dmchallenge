#!/bin/sh

source config &&
docker login ${dockerurl} &&
docker push ${preprocessimage}
