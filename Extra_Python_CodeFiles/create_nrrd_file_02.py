'''
This program creates a 3D density field in NRRRD format for use in ray tracing.

'''

import nrrd
import numpy as np
import matplotlib.pyplot as plt
import os
from sys import argv
import scipy.special

def save_nrrd(data,nrrd_filename):
# This function saves the density stored in data['rho'] to an nrrd file
# specified by nrrd_filename

    # specify orientation of space
    space = '3D-right-handed'

    # generate arrays to store co-ordinates along the three axes
    x = np.array(data['x'])
    y = np.array(data['y'])
    z = np.array(data['z'])

    # set origin
    x0 = x.min()
    y0 = y.min()
    z0 = z.min()

    space_orig = np.array([x0, y0, z0]).astype('float32')

    # set grid spacing
    del_x = np.diff(x)[0]
    del_y = np.diff(y)[0]
    del_z = np.diff(z)[0]

    spacings = np.array([del_x, del_y, del_z]).astype('float32')

    # spcify other relevant options
    options = {'type' : 'f4', 'space': space, 'encoding': 'raw',
               'space origin' : space_orig, 'spacings' : spacings}

    print ("saving density to %s" % nrrd_filename)

    # save data to nrrd file
    nrrd.write(nrrd_filename, np.array(data['rho']).astype('float32'), options)

def calculate_theoretical_deflection(grad_x, rho_0, M, Z_D, delta_z, pixel_pitch):
    '''
    This function calculates the theoretical displacement of a dot pattern on the camera sensor for a constant density
    gradient field using the paraxial approximation
    INPUTS:
    grad_x - density gradient field (kg/m^4)
    rho_0 - this is the density of the undisturbed medium (kg/m^3)
    M - magnification (no units)
    Z_D - distance betwen dot pattern and density gradient field (m)
    delta_z - thickness of the density gradient field (m)
    pixel_pitch - size of a pixel on the camera sensor (m)
    '''

    # this is the gladstone-dale constant (m^3/kg)
    K = 0.225e-3

    # this is the refractive index of the undisturbed medium
    n_0 = K * rho_0 + 1

    # this is the gradient of refractive index (m^-1)
    dn_dx = K * grad_x

    # angular deflection of light ray (radians)
    epsilon_x = 1/n_0 * dn_dx * delta_z

    # this is the displacement on the sensor (m)
    displacement = M * Z_D * epsilon_x

    print ('angular deflection of ray (radians): %.6G' % epsilon_x)
    print ('density gradient (kg/m^4): %.2G' % grad_x)
    print ('refractive index gradient (/m): %.4G' % dn_dx)
    print ('displacement on the sensor (mm): %.2G' % (displacement*1e3))
    print ('displacement on the sensor (pixels): %.2G' % (displacement/pixel_pitch))


def calculate_density_gradient(delta_x, delta_y, rho_0, M, Z_D, delta_z, pixel_pitch):
    '''
    this function calculates the density gradients required to produce the specified
    pixel displacement at the camera sensor given the extent of the density gradient
    volume, magnification and pixel pitch

    INPUTS:
    delta_x, delta_y: desired displacement of the dot pattern (pix.)
    data - file containing density field
    rho_0 - ambient density (kg/m^3)
    M - magnification (unitless)
    Z_D - distance between the dot pattern and the mid point of the density field (m)
    delta_z - thickness of the density gradient field (m)
    pixel_pitch - size of a pixel on the camera sensor (m)
    '''
    
    # this is the gladstone-dale constant (m^3/kg)
    K = 0.225e-3
    
    # this is the refractive index of the undisturbed medium
    n_0 = K * rho_0 + 1
    
    # this is the required angular deflection of the ray (radians)
    epsilon_x = delta_x * pixel_pitch/(M * Z_D)
    
    # this is the required refractive index gradient
    dn_dx = epsilon_x * n_0/delta_z
    
    # this is the required density gradient (kg/m^4)
    drho_dx = 1/K * dn_dx

    # this is the required gradient of density along x (kg/m^4)
    grad_x = 1/K * delta_x * pixel_pitch/(M * Z_D) * n_0/delta_z # * ((data['x'].max() - data['x'].min())*1e-6)

    # this is the required gradient of density along x (kg/m^4)
    grad_y = 1/K * delta_y * pixel_pitch/(M * Z_D) * n_0/delta_z # * ((data['y'].max() - data['y'].min())*1e-6)

    return grad_x, grad_y


def calculate_density_noise(displacement_noise_std, M, Z_D, delta_x, delta_z, rho_0, pixel_pitch):
    '''
    This function calculates the standard deviation of the noise to be added to the
    density field to obtain a specified standard deviation of the noise in the displacement
    field.
    
    This is calculated based on the propogation of errors in the BOS image generation methodology.
    
    It is assumed that the noise is a random variable drawn from a zero-mean Gaussian 
    distribution.
    
    INPUT:
    displacement_noise_std: standard deviation of required noise in the displacement field | scalar | float | pix.
    M: magnification of the optical system | scalar | float | unitless
    Z_D: distance between the target and the mid-point of the density gradient field | scalar | float | m
    delta_x: grid spacing along x for points in the density gradient field | scalar | float | m
    delta_z: extent of the density gradient field along z | scalar | float | m
    rho_0: ambient density | scalar | float | kg/m^3
    pixel_pitch: dimension of a single pixel on the camera sensor | scalar | float | m
    
    OUTPUT:
    rho_noise_std: standard deviation of noise to be added to the density field | scalar | float | kg/m^3
    '''
    
    # gladstone dale constant (m^3/kg)
    K = 0.225e-3
    
    # calculate ambient refractive index
    n_0 = K * rho_0 + 1
    
    # calculate standard deviation of noise to be added to the density field (added factor of 2 to account for central differencing in calculation of the refractive index gradient)
    rho_noise_std = 2 * displacement_noise_std * pixel_pitch * delta_x / (M * Z_D * K/n_0 * np.sqrt(2.0) * delta_z)
    
    return rho_noise_std


def create_nrrd_file(filepath, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, z_object, focal_length, pixel_pitch, mode, delta_x=0.0, delta_y=0.0, blur_factor=0.0, peak_density=0.0, std=0.0):
    '''
    This function creates a density gradient field and saves it to an NRRD format. 
    INPUTS:
    nx, ny, nz - array dimensions of the density gradient volume
    xmin, xmax - extents of the density field along x (m)
    ymin, ymax - extents of the density field along y (m)
    zmin, zmax - extents of the density field along z (m)        
    z_object - object distance (m)
    delta_z - thickness of the density gradient field (m)
    Z_D - distance between density gradient field and dot pattern (m)
    field_of_view - FOV for current optical layout (m)
    focal_length - camera focal length (m)
    pixel_pitch - camera pixel pitch (m)
    mode - 'linear' or 'quadratic'
    delta_x, delta_y - theoretical displacement along X and Y (pix.)
    blur_factor - second gradient of density (kg/m^5) in a 3 element array (xx, xy, yy).
                  A scalar input is assumed to represent the second gradient along xx.
    peak_density - peak density ( without including the ambient) for a gaussian distribution
    std - standard deviation of the density profile (m)
    '''

    data = {}

    # generate array of co-ordinates
    x = np.linspace(xmin, xmax,nx).astype('float32')
    y = np.linspace(ymin, ymax,ny).astype('float32')
    z = np.linspace(zmin, zmax, nz).astype('float32')

    X,Y = np.meshgrid(x,y,indexing='ij')
    
    # store array of co-ordinates
    data['x'] = x * 1e6
    data['y'] = y * 1e6
    data['z'] = z * 1e6
    
    # this is the density of the undisturbed medium (kg/m^3)
    rho_0 = 1.225


    # initialize array of densities
    data['rho'] = rho_0*np.ones([nx,ny,nz])

    # calculate the corresponding values of grad_x and grad_y
    if mode == 'linear':
        # thickness of the density gradient field
        delta_z = np.abs(zmax - zmin)
        # distance of the mid point of the density gradient field from the dot pattern
        Z_D = np.abs(z_object - (zmin + zmax)/2.0)
        # magnification
        M = focal_length/(z_object-focal_length)   
        # calculate gradient required with the given optical layout to produce the desired displacement of the dot pattern
        [grad_x, grad_y] = calculate_density_gradient(delta_x, delta_y, rho_0, M, pixel_pitch, delta_z, Z_D)
        # create density field
        for k in range(0,nz):
            data['rho'][:,:,k] += grad_x * (xmax - xmin) + grad_y * (ymax - ymin)
        # name of the file containing the density field
        # nrrd_filename = os.path.join(filepath, 'const_grad_delta_x=%.2f_delta_y=%.2f_zobj=%d_zmin=%d_zmax=%d_nx=%04d_ny=%04d_nz=%04d.nrrd' % (delta_x, delta_y, z_object*1e3, zmin*1e3, zmax*1e3, nx, ny, nz))        
        nrrd_filename = os.path.join(filepath, 'test-density.nrrd')
        # # calculate theoretical deflection given the optical layout and the density gradient field (should match the input deflection)
        # calculate_theoretical_deflection(data, grad_x, zmin, zmax, M, pixel_pitch)

    elif mode == 'quadratic':
        # if the blur factor is not an array, then set it to be the second gradient
        # of density along x. this is done to make the code backwards compatible.
        if not isinstance(blur_factor, list):
            blur_factor = [blur_factor, 0, 0]
        for k in range(0,nz):
            data['rho'][:,:,k] += blur_factor[0] * (X - x.min())**2 + blur_factor[1] * (X - x.min()) * (Y - y.min()) + blur_factor[2] * (Y - y.min())**2
            # nrrd_filename = os.path.join(top_nrrd_filepath, 'blur=%.2f_zobj=%d_zmin=%d_zmax=%d_nx=%04d_ny=%04d_nz=%04d.nrrd' % (blur_factor, z_object/1e3, z_1/1e3, z_2/1e3, nx, ny, nz))
        #nrrd_filename = os.path.join(filepath, 'blur-xx=%.2f-xy=%.2f-yy=%.2f_zobj=%d_zmin=%d_zmax=%d_nx=%04d_ny=%04d_nz=%04d.nrrd' % (blur_factor[0], blur_factor[1], blur_factor[2], z_object*1e3, zmin*1e3, zmax*1e3, nx, ny, nz))
        nrrd_filename = os.path.join(filepath, 'test-density.nrrd')

    elif mode == 'gaussian':
        # create a gaussian density profile
        for k in range(0,nz):
            data['rho'][:,:,k] += peak_density * np.exp(-1/(2*std**2) * ((X - np.mean(x))**2 +  (Y - np.mean(y))**2))
        # nrrd_filename = os.path.join(filepath, 'peak=%.2f_std=%.2fmm_xmin=%dmm_xmax=%dmm_ymin=%dmm_ymax=%dmm_zmin=%dmm_zmax=%dmm_nx=%04d_ny=%04d_nz=%04d.nrrd' % (peak_density, std*1e3, xmin*1e3, xmax*1e3, ymin*1e3, ymax*1e3, zmin*1e3, zmax*1e3, nx, ny, nz))
        nrrd_filename = os.path.join(filepath, 'test-density.nrrd')

    elif mode == 'erf':
        # create a gaussian density profile
        for k in range(0,nz):
            data['rho'][:,:,k] += peak_density * scipy.special.erf(1/(np.sqrt(2.0)*std) * (X - np.mean(x)))
        # nrrd_filename = os.path.join(filepath, 'erf_peak=%.2f_std=%.2fmm_xmin=%dmm_xmax=%dmm_ymin=%dmm_ymax=%dmm_zmin=%dmm_zmax=%dmm_nx=%04d_ny=%04d_nz=%04d.nrrd' % (peak_density, std*1e3, xmin*1e3, xmax*1e3, ymin*1e3, ymax*1e3, zmin*1e3, zmax*1e3, nx, ny, nz))
        nrrd_filename = os.path.join(filepath, 'test-density.nrrd')

    save_nrrd(data,nrrd_filename)
    return nrrd_filename
