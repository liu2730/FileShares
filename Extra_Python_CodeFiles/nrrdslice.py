import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import nrrd
import numpy as np

data, header = nrrd.read('/home/liu_fl/Downloads/photon-master/sample-data/bos/test-density.nrrd')
numel = np.size(data)
ndim = np.shape(data)
print('Number of elements: ', numel)
print('Array dimensions: ', ndim)


xslice = data[round(ndim[0]/2), :, :]
print('xslice numbers:', np.size(xslice), np.shape(xslice))
yslice = data[:, round(ndim[1]/2), :]
print('yslice numbers:', np.size(yslice), np.shape(yslice))
zslice = data[:, :, round(ndim[2]/2)]
print('zslice numbers:', np.size(zslice), np.shape(zslice))

#yz plane
x1 = np.arange(xslice.shape[0])
y1 = np.arange(xslice.shape[1])
fig1, ax1 = plt.subplots()
ax1.pcolormesh(x1, y1, xslice)
plt.savefig('/home/liu_fl/Downloads/photon-master/sample-data/xslice_newGauss_nrrd.png')

#xz plane
x2 = np.arange(yslice.shape[0])
y2 = np.arange(yslice.shape[1])
fig2, ax2 = plt.subplots()
ax2.pcolormesh(x2, y2, yslice)
plt.savefig('/home/liu_fl/Downloads/photon-master/sample-data/yslice_newGauss_nrrd.png')

#xy plane
x3 = np.arange(zslice.shape[0])
y3 = np.arange(zslice.shape[1])
fig3, ax3 = plt.subplots()
ax3.pcolormesh(x3, y3, zslice)
plt.savefig('/home/liu_fl/Downloads/photon-master/sample-data/zslice_newGauss_nrrd.png')