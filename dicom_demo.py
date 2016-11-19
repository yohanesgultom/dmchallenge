# Accessing DCM image properties
import dicom
import cv2
import numpy as np
import warnings
import matplotlib.pyplot as plt
from dicom.datadict import all_names_for_tag


# center crop non-zero and downsample to EXPECTED_SIZE
def center_crop_resize_filter(dat, expected_size, max_value):
    cropped = crop(dat)
    start = cropped.shape[0] / 2 - (cropped.shape[1] / 2)
    end = start + cropped.shape[1]
    cropped = cropped[start:end, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resized = cv2.resize(cropped, (expected_size, expected_size))
    assert resized.shape == (expected_size, expected_size)
    filtered = cv2.medianBlur(resized, 5)
    norm = filtered * 1.0 / max_value
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


# load dicom
negative_dcm = dicom.read_file("100151.dcm")
positive_dcm = dicom.read_file("100152.dcm")

# # get header info
# print(positive_dcm)

# get image pixels numpy array
p = crop(positive_dcm.pixel_array)
n = crop(negative_dcm.pixel_array)
pt = center_crop_resize_filter(positive_dcm.pixel_array, 224, 4095)
nt = center_crop_resize_filter(negative_dcm.pixel_array, 224, 4095)

# normalize
p = p / 4095.0
n = n / 4095.0

pt[pt < 0.4] = 0
nt[nt < 0.4] = 0

print(p.shape)
print(n.shape)
print(pt.shape)
print(nt.shape)

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
