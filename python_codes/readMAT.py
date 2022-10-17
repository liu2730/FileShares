import scipy.io as sio

#/home/liu_fl/Downloads/photon-master/sample-data/bos/parameters/sample-parameters.mat

matContents = sio.loadmat('/home/liu_fl/Downloads/photon-master/sample-data/bos/parameters/sample-parameters.mat')
matContents_ex = sio.loadmat('/home/liu_fl/Downloads/10_4231_P45Z-8361/bundle/analysis-package/jhu/images/t=8.00_zoff=0.75pi/1/parameters.mat')

print(matContents['__header__'])
print(matContents_ex['__header__'])