'''
Build DM dataset
Usage: python preprocess.py <in:DM images directory> <in:DM crosswalk file> <out:dataset h5>

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
import tables
import warnings
# from sklearn.preprocessing import StandardScaler

# expected width/length (assumed square)
EXPECTED_SIZE = 224
EXPECTED_CHANNELS = 3
EXPECTED_DIM = (EXPECTED_CHANNELS, EXPECTED_SIZE, EXPECTED_SIZE)
EXPECTED_CLASS = 1
MAX_VALUE = 4095

# static
# scaler = StandardScaler()


# preprocess image and return vectorized value
def preprocess_image(filename):
    dcm = dicom.read_file(filename)
    m = center_crop_resize(dcm.pixel_array)
    return np.array([[m, m, m]])


# center crop non-zero and downsample to EXPECTED_SIZE
def center_crop_resize(dat):
    # crop zeros (black parts)
    cropped = crop(dat)
    # center crop:
    # crop by height (y axis) to match width (x axis)
    start = cropped.shape[0] / 2 - (cropped.shape[1] / 2)
    end = start + cropped.shape[1]
    cropped = cropped[start:end, :]
    # resize (downsample)
    scale = EXPECTED_SIZE * 1.0 / cropped.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resized = scipy.ndimage.interpolation.zoom(cropped, scale, order=3, prefilter=True)
        assert resized.shape == (EXPECTED_SIZE, EXPECTED_SIZE)
        # scaled to 0..1
        norm = resized * 1.0 / MAX_VALUE
        # # normalize
        # norm = scaler.fit_transform(resized)
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

if __name__ == '__main__':
    dcm_dir = sys.argv[1]
    crosswalk_file = sys.argv[2]
    outfile = sys.argv[3]

    # pytables file
    datafile = tables.open_file(outfile, mode='w')
    data = datafile.create_earray(datafile.root, 'data', tables.Float32Atom(shape=EXPECTED_DIM), (0,), 'dream')
    labels = datafile.create_earray(datafile.root, 'labels', tables.UInt8Atom(shape=(EXPECTED_CLASS)), (0,), 'dream')

    # read crosswalk
    filenames = []
    with open(crosswalk_file, 'rb') as tsvin:
        crosswalk = csv.reader(tsvin, delimiter='\t')
        headers = next(crosswalk, None)
        for row in crosswalk:
            dcm_filename = row[5]
            dcm_label = int(row[6])
            filenames.append(dcm_filename)
            labels.append(np.array([[dcm_label]]))

    # read dicom images
    bar = progressbar.ProgressBar(maxval=len(filenames)).start()
    for i, dcm_filename in enumerate(filenames):
        data.append(preprocess_image(os.path.join(dcm_dir, dcm_filename)))
        bar.update(i)
    bar.finish()

    print(data[:].shape)
    print(labels[:].shape)

    # close file
    datafile.close()
