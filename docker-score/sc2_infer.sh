#!/usr/bin/env sh

# /inferenceData/*.dcm
# /metadata/images_crosswalk.tsv
# /metadata/exams_metadata.tsv (SC2 only)
# /modelState: the directory that contains the trained model
# /scratch: partition of 200 GB is available for pre-processing, training and scoring. All the files present in /scratch are removed before starting a new Docker container.
# /output: output directory

/usr/bin/time python sc2_infer.py /inferenceData /scratch /metadata/images_crosswalk.tsv /metadata/exams_metadata.tsv /modelState/model.arch.json /modelState/model.weights.h5 /output/predictions.tsv
