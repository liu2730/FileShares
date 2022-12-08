import create_nrrd_file_02 as CNRRD
import nrrd
import numpy as np

data, header = nrrd.read('/home/liu_fl/Downloads/photon-master/sample-data/bos/sample-density.nrrd')
print(data)
new_data = data * 10000
print(new_data)
nrrd.write('/home/liu_fl/Downloads/photon-master/sample-data/bos/test-density.nrrd', new_data, header)
