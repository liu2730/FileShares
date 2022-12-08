import light_ray_processing as LRP
import numpy as np
import sys
import math
from timeit import default_timer as timer

#np.set_printoptions(threshold=sys.maxsize)
start = timer()

pos1, pos2, dir1, dir2 = LRP.load_light_ray_data('/home/liu_fl/Downloads/photon-master/sample-data/bos/images')

data_transform = timer()

print('Seconds to organize data: ', (data_transform - start))

pos1 = [pos1['x'], pos1['y']]
pos2 = [pos2['x'], pos2['y']]

pos1temp = [pos1[0][0:2000], pos1[1][0:2000]]
pos2temp = [pos2[0][0:2000], pos2[1][0:2000]]
    

#Calculating pixel displacement
'''
pos1x = pos1['x']
pos1y = pos1['y']
pos2x = pos2['x']
pos2y = pos2['y']
displaceTempx = 0
displaceTempy = 0
displaceCount = 0
pixelDiff = []

for x in range(np.size(pos1x) - 1):
    displaceTempx = (pos2x[x] - pos1x[x]) / 17
    displaceTempy = (pos2y[x] - pos1y[x]) / 17
    totalDisplace = math.sqrt(displaceTempx**2 + displaceTempy**2)
    #pixelDiff.append(displaceTemp)
    if abs(totalDisplace) > 1:
        displaceCount = displaceCount + 1

print('Displacements greater than 1 pixel: ', displaceCount)

end = timer()

print('Total time taken (sec): ', (end - start))
'''


#Calculating number of different coordinates
'''
pos1coord = []
pos2coord = []
diffCounter = 0

for x in range(0, (np.size(pos1[0]) - 1)):
    pos1coord = [pos1[0][x], pos1[1][x], pos1[2][x]]
    pos2coord = [pos2[0][x], pos2[1][x], pos2[2][x]]
    if (pos1coord != pos2coord):
        diffCounter = diffCounter + 1

print('Number of Coord differences: ', diffCounter)
'''

#Writing to File

with open("/home/liu_fl/Downloads/photon-master/python_codes/pos1.txt", 'w') as f:
    for x in range(0, (np.size(pos1temp[0]) - 1)):
        f.write(f"{pos1temp[0][x]}, {pos1temp[1][x]}\n")
    f.close()

with open("/home/liu_fl/Downloads/photon-master/python_codes/pos2.txt", 'w') as f:
    for x in range(0, (np.size(pos1temp[0]) - 1)):
        f.write(f"{pos2temp[0][x]}, {pos2temp[1][x]}\n")
    f.close()
