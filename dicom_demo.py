# Accessing DCM image properties

import dicom
import numpy as np
import pylab
import warnings
import matplotlib.pyplot as plt
from dicom.datadict import all_names_for_tag
from scipy import ndimage


# center crop non-zero and downsample to EXPECTED_SIZE
def center_crop_resize(dat, expected_size, max_value):
    cropped = crop(dat)
    start = cropped.shape[0] / 2 - (cropped.shape[1] / 2)
    end = start + cropped.shape[1]
    cropped = cropped[start:end, :]
    scale = expected_size * 1.0 / cropped.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resized = ndimage.interpolation.zoom(cropped, scale, order=3, prefilter=True)
        assert resized.shape == (expected_size, expected_size)
        norm = resized * 1.0 / max_value
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


def plot(images):
    fig = pylab.figure()
    for i, m in enumerate(images):
        fig.add_subplot(len(images), 1, i + 1)
        pylab.imshow(m, cmap=pylab.cm.bone)
    pylab.show()

# load dicom
negative_dcm = dicom.read_file("100151.dcm")
positive_dcm = dicom.read_file("100152.dcm")

# # get header info
# print(positive_dcm)

# get image pixels numpy array
p = crop(positive_dcm.pixel_array)
n = crop(negative_dcm.pixel_array)
pt = center_crop_resize(positive_dcm.pixel_array, 224, 4095)
nt = center_crop_resize(negative_dcm.pixel_array, 224, 4095)

# normalize
p = p / 4095.0
n = n / 4095.0

pt[pt < 0.4] = 0
nt[nt < 0.4] = 0

# plt.matshow(ndimage.gaussian_filter(pt, 2), 1)
# plt.matshow(ndimage.gaussian_filter(nt, 2), 2)
plt.matshow(ndimage.median_filter(pt, 4), 3)
plt.matshow(ndimage.median_filter(nt, 4), 4)
plt.show()

# # accessing info
# elem = postive_dcm[0x0010, 0x0010]
# print(elem.tag, elem.description(), elem.value)
