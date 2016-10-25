# Accessing DCM image properties

import dicom
from dicom.datadict import all_names_for_tag
import numpy as np
import pylab

# load dicom
ds = dicom.read_file("000135.dcm")

# get header info
print(ds)
print("\n")

# get image pixels numpy array
print(ds.pixel_array)
print(ds.pixel_array.shape)
print("\n")

# accessing info
elem = ds[0x0010, 0x0010]
print(elem.tag, elem.description(), elem.value)

# render image
pylab.imshow(ds.pixel_array, cmap=pylab.cm.bone)
pylab.show()
