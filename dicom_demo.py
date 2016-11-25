# Accessing DCM image properties
import dicom
import cv2
import numpy as np
import warnings
import matplotlib.pyplot as plt
from dicom.datadict import all_names_for_tag

EXPECTED_MAX = 100
EXPECTED_MIN = -1 * EXPECTED_MAX
FILTER_THRESHOLD = -90.0

EXPECTED_SIZE = 224
MAX_VALUE = 4095.0
MEDIAN_VALUE = MAX_VALUE / 2.0  # 0..MAX_VALUE


# center crop non-zero and downsample to EXPECTED_SIZE
def center_crop_resize_filter(dat, laterality, median, expected_min, expected_max, expected_size, filter_threshold):
    res = crop(dat)
    start = res.shape[0] / 2 - (res.shape[1] / 2)
    end = start + res.shape[1]
    res = res[start:end, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = cv2.resize(res, (expected_size, expected_size))
    assert res.shape == (expected_size, expected_size)
    # res = cv2.medianBlur(res, 5)
    res = (res - median) / median * expected_max
    if laterality.upper() == 'R':
        res = np.fliplr(res)
    print(res.shape, np.amin(res), np.amax(res))
    res[res < filter_threshold] = expected_min
    return res


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


# load dicom
# negative_dcm = dicom.read_file("100151.dcm")  # Right
# positive_dcm = dicom.read_file("100152.dcm")  # Left
positive_dcm = dicom.read_file("100153.dcm")  # Left
negative_dcm = dicom.read_file("100154.dcm")  # Right

# # get header info
# print(positive_dcm)

# get image pixels numpy array
p = positive_dcm.pixel_array
n = negative_dcm.pixel_array

pt = center_crop_resize_filter(positive_dcm.pixel_array, "L", MEDIAN_VALUE, EXPECTED_MIN, EXPECTED_MAX, EXPECTED_SIZE, FILTER_THRESHOLD)
nt = center_crop_resize_filter(negative_dcm.pixel_array, "R", MEDIAN_VALUE, EXPECTED_MIN, EXPECTED_MAX, EXPECTED_SIZE, FILTER_THRESHOLD)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
ax1.imshow(p)
ax2.imshow(pt)
ax3.imshow(n)
ax4.imshow(nt)
plt.tight_layout()
plt.show()

# # accessing info
# elem = postive_dcm[0x0010, 0x0010]
# print(elem.tag, elem.description(), elem.value)
