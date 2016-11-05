'''
Build DM dataset
Usage: python preprocess.py <DM images directory> <DM crosswalk file> <output file>

'''
import dicom
import os
import numpy
import sys
import numpy as np
# import pylab
import csv
import scipy.ndimage.interpolation
import progressbar

# expected width/length (assumed square)
EXPECTED_SIZE = 224
MAX_VALUE = 4095


# center crop non-zero and downsample to EXPECTED_SIZE
def center_crop_resize(dat):
    # remove zeros
    cropped = crop(dat)
    # center crop
    start = cropped.shape[0] / 2 - (cropped.shape[1] / 2)
    end = start + cropped.shape[1]
    cropped = cropped[start:end, :]
    # resize (downsample)
    scale = EXPECTED_SIZE * 1.0 / cropped.shape[1]
    resized = scipy.ndimage.interpolation.zoom(cropped, scale, order=3, prefilter=True)
    # scaled to 0..1
    norm = resized * 1.0 / MAX_VALUE
    # print(cropped.shape)
    # print(resized.shape)
    # print(resized)
    # print(norm)
    # # render image
    # images = [dat, norm]
    # fig = pylab.figure()
    # for i, m in enumerate(images):
    #     fig.add_subplot(len(images), 1, i + 1)
    #     pylab.imshow(m, cmap=pylab.cm.bone)
    # pylab.show()
    return norm


# Crop non-zero rectangle
# Original http://stackoverflow.com/questions/39465812/how-to-crop-zero-edges-of-a-numpy-array
def crop(dat):
    # argwhere will give you the coordinates of every non-zero point
    true_points = np.argwhere(dat)
    # take the smallest points and use them as the top left of your crop
    top_left = true_points.min(axis=0)
    # take the largest points and use them as the bottom right of your crop
    bottom_right = true_points.max(axis=0)
    # plus 1 because slice isn't inclusive
    return dat[
        top_left[0]:bottom_right[0] + 1,
        top_left[1]:bottom_right[1] + 1
    ]


dcm_dir = sys.argv[1]
crosswalk_file = sys.argv[2]
outfile = sys.argv[3]

# read crosswalk
labels = []
filenames = []
with open(crosswalk_file, 'rb') as tsvin:
    crosswalk = csv.reader(tsvin, delimiter='\t')
    headers = next(crosswalk, None)
    for row in crosswalk:
        dcm_filename = row[5]
        dcm_label = int(row[6])
        filenames.append(dcm_filename)
        labels.append(dcm_label)

# read dicom images
data = []
bar = progressbar.ProgressBar(max_value=len(filenames))
for i, dcm_filename in enumerate(filenames):
    dcm = dicom.read_file(os.path.join(dcm_dir, dcm_filename))
    m = center_crop_resize(dcm.pixel_array)
    data.append(np.array([m, m, m]))  # mimic 3 channel
    bar.update(i)
bar.finish()
x = np.array(data)
y = np.array(labels)
np.savez(outfile, x=x, y=y)
