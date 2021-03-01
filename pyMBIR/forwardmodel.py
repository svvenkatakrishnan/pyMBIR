# Copyright (C) 2019, S.V.Venkatakrishnan <venkatakrisv@ornl.gov>
# All rights reserved. GPL v3 license. 
# This file is part of the pyMBIR package. Details of the copyright
# and user license can be found in the 'LICENSE' file distributed
# with the package.

import numpy as np
from scipy.signal import fftconvolve
from scipy.fftpack import fft2,ifft2,fftshift

def computeGenHuberCost(error,weight_data,sigma,huber_T,huber_delta):
    """
    Function to compute cost function value corresponding to the generalized Huber function 
    """
    weight_mask_1 = np.where((np.fabs(error)*np.sqrt(weight_data)/sigma)<huber_T)
    weight_mask_2 = np.where((np.fabs(error)*np.sqrt(weight_data)/sigma)>=huber_T)
    cost_val = 0
    cost_val+= (0.5*((error[weight_mask_1]**2)*weight_data[weight_mask_1]).sum())/(sigma**2)
    cost_val+= (0.5*(2*huber_delta*huber_T*np.fabs(error[weight_mask_2]*np.sqrt(weight_data[weight_mask_2])/sigma)).sum() + 0.5*(huber_T**2)*(1-2*huber_delta)*(np.float64(weight_mask_2[0].size)))
    cost_val+=(0.5*np.float64(weight_data.size)*np.log(sigma**2))
    return cost_val

def computeGenHuberWeight(error,sigma,huber_T,huber_delta,weight_data,weight_new):
    """
    Function to compute updated weight matrix for the majorizer of the generalized Huber function 
    """
    weight_mask = np.where((np.fabs(error)*np.sqrt(weight_data)/sigma)>=huber_T)
#    print('Fraction of measurements being rejected = %f' % (np.float64(weight_mask[0].size)/weight_data.size))
    #Re-weighting
    np.copyto(weight_new,weight_data)
    weight_new[weight_mask]=np.sqrt(weight_data[weight_mask])*(huber_delta*huber_T*sigma/np.fabs(error[weight_mask]))
    return weight_new

def computeTalwarWeight(error,sigma,huber_T,huber_delta,weight_data,weight_new):
    """
    Function to compute updated weight matrix for the majorizer for the clipped Huber/Talwar penalty 
    TODO: Remove the huber_delta paramater 
    """
    #Compute updated weight matrix based on previous error
    weight_mask = np.where((np.fabs(error)*np.sqrt(weight_data)/sigma)>=huber_T)
#    print('Fraction of measurements being rejected = %f' % (np.float64(weight_mask[0].size)/weight_data.size))
    #Re-weighting
    np.copyto(weight_new,weight_data)
    weight_new[weight_mask]=0
    return weight_new

def LipschitzForward(obj_size,A,weight_data):
    """
      Function to compute an upper bound of Lipschitz constant of the forward model term 
      Input: 
             obj_size: A 3 element list  
             A : Tomographic projection matrix (Astra spot operator)
             W : Inv. covariance weight matrix 
      Output: 
             L = Max value of diag(A^{T}*W*A)
    """
    x_ones = np.ones(obj_size,dtype=np.float32)
    temp_proj = A*x_ones
    temp_proj*=weight_data
    temp_backproj = A.T*temp_proj
    return temp_backproj

def powerIter(obj_size,A,weight_data,num_iter):
    """
      Function to compute max Eigen value of matrix A^{T}WA using the power iteration method 
      Input: 
             obj_size: A 3 element list  
             A : Tomographic projection matrix (Astra spot operator)
             W : Inv. covariance weight matrix 
      Output: 
             L = max eigen value
    """
    x_r1 = np.ascontiguousarray(np.random.rand(obj_size).astype(np.float32))
    temp_proj = A*x_r1
    temp_proj = temp_proj*weight_data
    x_r2 = A.T*temp_proj
    for _iter in range(num_iter):
        x_r2n = np.linalg.norm(x_r2)
        x_r1 = x_r2/x_r2n
        temp_proj = A*x_r1
        temp_proj = temp_proj*weight_data
        x_r2 = A.T*temp_proj
        eig_val = np.dot(x_r1,x_r2)
        #print('Iter %d Current eig val estimate %f' % (_iter,eig_val))
        
    return x_r1,eig_val

def forwardProject(A,H,x,proj_shape):
    """
      Function to forward project a volume and blur it by a shift-invariant kernel H for each view 
      Input: 
             A : Tomographic projection matrix (Astra spot operator)
             H : FFT of Blur  kernel of size num_proj X num_row X num_col 
             x : Inputs volume 
      Output: 
             y : H*A*x projection
    """
    y1 = A*x
    y1 = y1.reshape(proj_shape)
    y2 = np.zeros(proj_shape)
    for idx in range(proj_shape[1]):
        y2[:,idx,:]=np.real((ifft2((fft2(np.squeeze(y1[:,idx,:]),[proj_shape[0],proj_shape[2]]))*H[idx])))
    y2=y2.reshape(np.prod(np.array(proj_shape)))
    return y2

def backProject(A,H,y,proj_shape):
    """
      Function to backproject (view dependent linear shift invariant blur + tomographic op. transpose) 
      Input: 
             A : Tomographic projection matrix (Astra spot operator)
             H : FFT of Blur  kernel of size  num_proj X num_row X num_col 
             y : Input
      Output: 
             y : A^T * H^T * y  projection
    """
    y2 = np.ascontiguousarray(np.zeros(proj_shape).astype(np.float32))
    y = y.reshape(proj_shape)
    for idx in range(proj_shape[1]):
        y2[:,idx,:]=np.real((ifft2((fft2(np.squeeze(y[:,idx,:]),shape=[proj_shape[0],proj_shape[2]]))*np.conj(H[idx]))))
    y2=y2.reshape(np.prod(np.array(proj_shape)))
    x = A.T*y2
    return x 


def LipschitzForwardBlurTomo(obj_size,proj_shape,A,H,weight_data):
    """
      Function to compute an upper bound of the Lipschitz constant of 0.5*||y-HAx||_W^{2}
      Input: 
             A : Tomographic projection matrix (Astra spot operator)
             H : FFT of Blur  kernel of size  num_row X num_col 
             W : Inv. covariance weight matrix 
      Output: 
             L : A^{T}H^{T}WHA
    """
    x_ones = np.ones(obj_size,dtype=np.float32)
    temp_proj = forwardProject(A,H,x_ones,proj_shape)
    temp_backproj = backProject(A,H,temp_proj*weight_data,proj_shape)
    return temp_backproj

def powerIterBlurTomo(obj_size,proj_shape,A,H,weight_data,num_iter):
    """
      Function to compute max Eigen value of matrix A^{T}WA using the power iteration method 
      Input: 
             obj_size: Number of elements in 3D volume  
             proj_shape : Shape of projection data 
             A : Tomographic projection matrix (Astra spot operator)
             H : FFT of Blur  kernel of size  num_proj X num_row X num_col 
             W : Inv. covariance weight matrix 
      Output: 
             L = max eigen value
    """
    x_r1 = np.ascontiguousarray(np.random.rand(obj_size).astype(np.float32))
    temp_proj = forwardProject(A,H,x_r1,proj_shape)
    temp_proj = temp_proj*weight_data
    #print('Min and max of proj of random input %f %f' %(temp_proj.min(),temp_proj.max()))
    x_r2 = backProject(A,H,temp_proj,proj_shape)
    #print('Min and max of back proj of random input %f %f' %(x_r2.min(),x_r2.max()))
    for _iter in range(num_iter):
        x_r2n = np.linalg.norm(x_r2)
        x_r1 = x_r2/x_r2n
        temp_proj = forwardProject(A,H,x_r1,proj_shape)
        temp_proj = temp_proj*weight_data
        x_r2 = backProject(A,H,temp_proj,proj_shape)
        eig_val = np.dot(x_r1,x_r2)
        #print('Iter %d Current eig val estimate %f' % (_iter,eig_val))
        
    return x_r1,eig_val
