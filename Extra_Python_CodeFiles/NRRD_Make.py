
import create_nrrd_file_02 as CNRRD
import nrrd
import numpy as np

filepath = '/home/liu_fl/Downloads/photon-master/sample-data/bos/'
rho_0 = 1.225 #Ambient Density (kg/m^3)

#Array Dimensions of Density volume
nx = 100
ny = 100
nz = 100

#Volume of Gradient Density (m) (keep these in meters!)
xmin = -0.1
xmax = 0.1
ymin = -0.1
ymax = 0.1
zmin = -0.5
zmax = 0.5
#Object distance (m)
z_object = 10 #50

#Simulation/Camera Parameters (in meters in original file) (microns in parameters)
focal_length = 0.105 #m
pixel_pitch = 17e-6 #m 

#Choose between linear, quadratic, gaussian, and erf
mode = 'gaussian'

#Theoretical Displacements (pixels)
delta_x=0
delta_y=0

#For Gaussian Calculations
blur_factor=0 #For Quadratic
peak_density= 1 * rho_0
std = 0.05

filename = CNRRD.create_nrrd_file(filepath, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, z_object, focal_length, pixel_pitch, mode, delta_x, delta_y, blur_factor, peak_density, std)

print (filename)
