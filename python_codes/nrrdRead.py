import numpy as np
import nrrd

filename = '/home/liu_fl/Downloads/photon-master/sample-data/bos/sample-density.nrrd'

readdata, header = nrrd.read(filename)
print(readdata.shape)
print(header)


#'jhu_buoy_turb_t=8.00_zoff=0.75pi'

filename = '/home/liu_fl/Downloads/photon-master/sample-data/bos/jhu_buoy_turb_t=8.00_zoff=0.75pi.nrrd'

readdata, header = nrrd.read(filename)
print(readdata.shape)
print(header)