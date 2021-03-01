# Copyright (C) 2019, S.V.Venkatakrishnan <venkatakrisv@ornl.gov>
# All rights reserved. GPL v3 license.
# This file is part of the pyMBIR package. Details of the copyright
# and user license can be found in the 'LICENSE' file distributed
# with the package.

import numpy as np
import psutil
import concurrent.futures as cf
from scipy.ndimage.interpolation import shift
from scipy.ndimage import rotate 
from numpy.fft import fftshift

#Useful utilities for simulation, testing etc.
def createTransmission(proj,I0,noise_std):
    """Function to generate the noisy projection data based on Beer's law and Gaussian approximation to Poisson statistics 
    Inputs: proj: An array containing the projections 
            I0 : The input flux (counts/pixel)
            noise_std: Noise standard deviation factor (scalar, 1 -> Poisson)
    Output : noisy_proj : Numpy array containg noisy projections
    """
    temp_proj = I0*np.exp(-proj)
    noisy_proj = temp_proj + noise_std*np.sqrt(temp_proj)*np.random.randn(*temp_proj.shape)
    return noisy_proj

def rmse(a,b):
    '''
    Root mean squared error between two arrays 
    Inputs: a,b : numpy arrays 
    Output: Root-mean squared error 
    '''
    #Root mean squared error between 2 arrays
    rmse = np.sqrt((((a-b)**2).sum())/a.size)
    return rmse

def stoppingCritVol(x_curr,x_prev,threshold,roi_mask):
    '''
    Function to compute relative change in the reconstruction to terminate the algorithm 
    Inputs: x_curr: Array of current reconstruction 
            x_prev: Array of the previous reconstruction 
            threshold: Stopping threshold 
            roi_mask : Binary mask of the same size as the x_curr and x_prev 
    Output: stop : Boolean variable based on if stopping criterial has been met 
    '''
    stop = False
    mask = roi_mask 
    abs_diff = np.fabs(x_curr[mask]-x_prev[mask])
    abs_prev = np.fabs(x_prev[mask])
    rel_change=100*abs_diff.mean()/abs_prev.mean()
#    print('Relative average change = %f percent'%((rel_change)))
    if(rel_change < threshold):
        stop = True
    return stop

def create_circle_mask(y,x,center,rad):
    '''
    Function to create a binary circular mask over a grid 
    '''
    mask = (x-center[0])*(x-center[0]) + (y-center[1])*(y-center[1]) <= rad*rad
    return mask

def create_sphere_mask(z,y,x,center,rad):
    '''
    Function to create a binary spherical mask over a grid 
    '''
    mask = (z-center[0])*(z-center[0]) + (y-center[1])*(y-center[1]) + (x-center[2])*(x-center[2]) <= rad*rad
    return mask 


def createGammaHits(data,view_frac,avg_hits_view,intensity,radius=3):
    '''
    Function to simulate outliers that may be encountered in various CT systems 
    '''
    num_rows,num_angles,num_cols=data.shape
    view_idx = np.sort(np.floor(np.random.rand(np.ceil(view_frac*num_angles/100))*(num_angles-1)))
    print(view_idx)
    for ang_list in view_idx:
        temp=np.random.rand(num_rows,1,num_cols)<=avg_hits_view/(num_rows*num_cols)
        idx_list=np.argwhere(temp==True)
        for elt in idx_list:
            data[elt[0]-radius:elt[0]+radius,ang_list,elt[2]-radius:elt[2]+radius]=intensity
    return data    
        
def genCTF(c1,c2,att_coeff,im_row,im_col,alpha=0):
    '''
    Generate CTF blur kernel; c2 = Cs*\lambda^3; c1 = \lambda*defocus alpha: Phase shift 
    '''
    v,u = np.ogrid[-im_row/2:im_row/2,-im_col/2:im_col/2]
    v=v/im_row
    u=u/im_col
    k = np.sqrt(v**2 + u**2)
    CTF = fftshift(np.exp(-att_coeff*k)*np.sin(-np.pi*c1*(k**2) + (np.pi/2)*c2*(k**4) + alpha))
    return CTF

def genCTF2(pix_size,voltage,defocus_u,defocus_v,defocus_ang,Cs,bfactor,amp,im_row,im_col):
    '''
    Generate CTF that matches the RELION software
    '''
    wav_len = np.sqrt(1.5/voltage) #nm
    v,u = np.ogrid[-im_row/2:im_row/2,-im_col/2:im_col/2]
    v/=(im_row*pix_size) #nm^-1
    u/=(im_col*pix_size)
    k = np.sqrt(v**2 + u**2)
    u[u==0]=1e-12
    theta = np.abs(np.arctan2(v,u))
    defocus_ang*=np.pi/180
    z_theta = defocus_u*(np.cos(theta-defocus_ang))**2 + defocus_v*(np.sin(theta-defocus_ang))**2
    c1 = wav_len*z_theta
    c2=Cs*(wav_len**3)
    gamma_k = -(np.pi/2)*c2*(k**4) + np.pi*c1*(k**2)
    amp_mat= amp*np.exp(-bfactor*k)
    amp_1 = np.sqrt(1.0-amp_mat**2)
    CTF = fftshift(-amp_mat*np.cos(gamma_k)-amp_1*np.sin(gamma_k))
    return CTF

def genGaussBlur(filt_size,im_row,im_col):
    '''
    Generate FFT of Gaussian Blur Kernel 
    '''
    h = np.zeros((filt_size,filt_size))
    h[filt_size//2,filt_size//2]=1
    from scipy.ndimage import gaussian_filter
    from scipy.fftpack import fft2
    h=gaussian_filter(h, sigma=filt_size/2)
    h = h/h.sum()
    fft_h = fft2(h,shape=[im_row,im_col])
    H=fft_h
    return H

def apply_proj_offsets(proj, proj_offsets, ncore=None, out=None):
    '''
    Code to apply offsets to projection data using an interpolation kernel 
    '''
    if not ncore:
      ncore = psutil.cpu_count(True)
    if out is None:
      out = np.empty(proj.shape, dtype=proj.dtype)
    with cf.ThreadPoolExecutor(ncore) as e:
      futures = [e.submit(shift, proj[i], proj_offsets[i], out[i], order = 1, mode='nearest') for i in range(proj.shape[0])]#nearest
      cf.wait(futures)
    return out

def apply_proj_tilt(proj, tilt_angle, ncore=None, out=None):
    '''
    Code to apply rotation to projection data to correct for tilts 
    '''
    if not ncore:
      ncore = psutil.cpu_count(True)
    if out is None:
      out = np.empty(proj.shape, dtype=proj.dtype)
    with cf.ThreadPoolExecutor(ncore) as e:
      futures = [e.submit(rotate, proj[i], tilt_angle, output=out[i],axes =(1,0),reshape=False,mode='nearest') for i in range(proj.shape[0])]#nearest
      cf.wait(futures)
    return out
