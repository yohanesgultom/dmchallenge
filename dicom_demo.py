# Accessing DCM image properties

import dicom
import numpy as np
import pylab
import cv2
import warnings
from dicom.datadict import all_names_for_tag
from sklearn.preprocessing import StandardScaler


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


def scaledown(src):
    src *= (255 / src)
    return cv2.convertScaleAbs(src)


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
positive_array_full = positive_dcm.pixel_array
positive_array = crop(positive_array_full)
# plot([positive_array_full, positive_array])

# histogram equalization
# positive_scaled = scaledown(positive.pixel_array)
# equalized = cv2.equalizeHist(src1)

# normalize
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    positive_array_normal = StandardScaler().fit_transform(positive_array)
    print(positive_array.max())
    print(positive_array_normal.max())

# # accessing info
# elem = postive_dcm[0x0010, 0x0010]
# print(elem.tag, elem.description(), elem.value)

# # render image
# pylab.imshow(ds.pixel_array, cmap=pylab.cm.bone)
# pylab.show()
