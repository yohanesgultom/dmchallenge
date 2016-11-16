#!/usr/bin/env sh

# The following volumes are mounted when the preprocessing container is run:
# - /trainingData (RO)
# - /metadata (RO)
# - /preprocessedData (RW)
# furthermore, /metadata contains the following files:
# - /metadata/exams_metadata.tsv
# - /metadata/images_crosswalk.tsv

# display host specs
echo $(nproc) CPUs available.
echo $(free -m)

# generate keras-compatible dataset
/usr/bin/time python preprocess.py "/trainingData" "/metadata/images_crosswalk.tsv" "/metadata/exams_metadata.tsv" "/preprocessedData/metadata.pickle" "/preprocessedData/dataset.h5"
