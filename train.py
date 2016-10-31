import numpy as np
import sys

outfile = sys.argv[1]
npzfile = np.load(outfile)
x = npzfile['x']
y = npzfile['y']
print(x.shape)
print(y.shape)
