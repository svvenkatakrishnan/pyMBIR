# Copyright (C) 2019, S.V.Venkatakrishnan <venkatakrisv@ornl.gov>
# All rights reserved. GPL v3 License.
# This file is part of the pyMBIR package. Details of the copyright
# and user license can be found in the 'LICENSE' file distributed
# with the package.

import numpy as np
from pyMBIR.geometry import generateAmatrix
from pyMBIR.reconstruction import *
from matplotlib import pyplot as plt

def analytic(proj_data,proj_params,miscalib,vol_params,rec_params):
    '''
    Wrapper function for analytic reconstruction based on differnet geometries 
    Inputs: proj_data: Array of size det_row X num views X det_col 
            proj_params: Dictionary of projection data parameters with keys relevant to the specific geometry (see examples for details)
            miscalib: Dictionary of miscalibration parameters relevant to the specific geometry (center of rotation, tilts etc.)
            vol_params: Dictionary with keys relevant to the geometry of the object to be reconstructed 
            rec_params: Dictionary with keys relevant to the reconstruction algorithm 
    Output: rec : Array containing reconstruction 
    '''
    proj_data = np.require(proj_data,dtype=np.float32,requirements=['A','C'])

    #Depending on type of geometry FBP or FDK or ..
    if(proj_params['type']=='par'):
        rec=fbpCuda(proj_data,proj_params,miscalib,vol_params,rec_params)
    elif (proj_params['type']=='cone'):
        rec=fdkCuda(proj_data,proj_params,miscalib,vol_params,rec_params)
    return rec

def SIRT(proj_data,proj_params,miscalib,vol_params,rec_params):
    '''
    Wrapper function for SIRT reconstruction for different geometries 
    Inputs: proj_data: Array of size det_row X num views X det_col 
            proj_params: Dictionary of projection data parameters with keys relevant to the specific geometry (see examples for details)
            miscalib: Dictionary of miscalibration parameters relevant to the specific geometry (center of rotation, tilts etc.)
            vol_params: Dictionary with keys relevant to the geometry of the object to be reconstructed 
            rec_params: Dictionary with keys relevant to the reconstruction algorithm 
    Output: rec : Array containing reconstruction 
    '''
    proj_data = np.require(proj_data,dtype=np.float32,requirements=['A','C'])
    A=generateAmatrix(proj_params,miscalib,vol_params,rec_params['gpu_index'])
    if(proj_params['type'] == 'cone'):
        pix_x=proj_params['cone_params']['pix_x']
        pix_y=proj_params['cone_params']['pix_y']
    elif (proj_params['type'] == 'par'):
        pix_x = proj_params['pix_x']
        pix_y = proj_params['pix_y']
    else:
        pix_x = proj_params['pix_x']
        pix_y = proj_params['pix_y']

    if 'vol_row' not in rec_params.keys() or 'vol_col' not in rec_params.keys(): 
        rec_params['n_vox_z']=vol_params['n_vox_z']  
        rec_params['n_vox_x']=vol_params['n_vox_x'] 
        rec_params['n_vox_y']=vol_params['n_vox_y']

    if 'verbose' not in rec_params.keys():
        rec_params['verbose']=False

    rec=sirtCudaopTomo(proj_data,A,rec_params)

    return rec

def MBIR(proj_data,weight_data,proj_params,miscalib,vol_params,rec_params):
    '''
    Wrapper function for MBIR reconstruction with different forward models and an MRF prior 
    Inputs: proj_data: Array of size det_row X num views X det_col containing projection data 
            weight_data : Array of size det_row X num views X det_col containing weight data 
            proj_params: Dictionary of projection data parameters with keys relevant to the specific geometry (see examples for details)
            miscalib: Dictionary of miscalibration parameters relevant to the specific geometry (center of rotation, tilts etc.)
            vol_params: Dictionary with keys relevant to the geometry of the object to be reconstructed 
            rec_params: Dictionary with keys relevant to the reconstruction algorithm 
    Output: recon : Array containing 3D reconstruction 
    '''
    DEFAULT_STOP_THRESH=.5 #percentage change at which to terminate algorithm 
    
    weight_data = np.require(weight_data,dtype=np.float32,requirements=['A','C'])
    weight_data/=weight_data.mean() #Normalize weight data for intuitive regularization 
    
    #Generate A matrix depending on geometry type
    A=generateAmatrix(proj_params,miscalib,vol_params,rec_params['gpu_index'])

    if(proj_params['type'] == 'cone'):
        pix_x=proj_params['cone_params']['pix_x']
        pix_y=proj_params['cone_params']['pix_y']
    elif (proj_params['type'] == 'par'):
        pix_x = proj_params['pix_x']
        pix_y = proj_params['pix_y']
    else:
        pix_x = proj_params['pix_x']
        pix_y = proj_params['pix_y']

    if 'n_vox_x' not in rec_params.keys() or 'vol_col' not in rec_params.keys(): 
        rec_params['n_vox_z']=vol_params['n_vox_z']  
        rec_params['n_vox_x']=vol_params['n_vox_x'] 
        rec_params['n_vox_y']=vol_params['n_vox_y']

    if 'stop_thresh' not in rec_params.keys():
        rec_params['stop_thresh'] = DEFAULT_STOP_THRESH

    if 'verbose' not in rec_params.keys():
        rec_params['verbose']=False
        
    temp_vol_z = rec_params['n_vox_z']
    temp_vol_y = rec_params['n_vox_y']
    temp_vol_x = rec_params['n_vox_x']

    ROI_MASK = np.zeros((temp_vol_z,temp_vol_y,temp_vol_x),dtype=np.bool)
    y,x=np.ogrid[-temp_vol_y/2:temp_vol_y/2,-temp_vol_x/2:temp_vol_x/2]
    temp_roi_mask=create_circle_mask(y,x,np.array([0,0]),temp_vol_x/2)
    ROI_MASK[:]=temp_roi_mask
    rec_params['roi_mask']=np.reshape(ROI_MASK,temp_vol_z*temp_vol_y*temp_vol_x)
    
    if(proj_params['forward_model_idx']==1):
        rec,cost=mbiropTomo(proj_data,A,rec_params)
    elif (proj_params['forward_model_idx']==2):
        rec,cost=mbiropTomoPoisson(proj_data,weight_data,A,rec_params)
    elif (proj_params['forward_model_idx']==3):
        rec,cost,ac=mbiropTomoTalwar(proj_data,weight_data,A,rec_params)
    elif (proj_params['forward_model_idx']==4):
        H = np.array(rec_params['H_blur'])
        rec,cost=mbiropDeblurTomoPoisson(proj_data,weight_data,A,H,rec_params)
    if(rec_params['debug']):
        plt.plot(cost);plt.xlabel('Iter');plt.ylabel('Cost function');plt.show()

    if 'ac' in locals():
        return rec,ac
    else:
        return rec


def ML(proj_data,weight_data,proj_params,miscalib,vol_params,rec_params):
    '''
    Wrapper function for maximum likelihood reconstruction with different forward models
    Inputs: proj_data: Array of size det_row X num views X det_col containing projection data 
            weight_data : Array of size det_row X num views X det_col containing weight data 
            proj_params: Dictionary of projection data parameters with keys relevant to the specific geometry (see examples for details)
            miscalib: Dictionary of miscalibration parameters relevant to the specific geometry (center of rotation, tilts etc.)
            vol_params: Dictionary with keys relevant to the geometry of the object to be reconstructed 
            rec_params: Dictionary with keys relevant to the reconstruction algorithm 
    Output: recon : Array containing 3D reconstruction 
    '''
    DEFAULT_STOP_THRESH=.5 #percentage change at which to terminate algorithm 
    
    weight_data = np.require(weight_data,dtype=np.float32,requirements=['A','C'])
    weight_data/=weight_data.mean() #Normalize weight data for intuitive regularization 
    
    #Generate A matrix depending on geometry type
    A=generateAmatrix(proj_params,miscalib,vol_params,rec_params['gpu_index'])

    if(proj_params['type'] == 'cone'):
        pix_x=proj_params['cone_params']['pix_x']
        pix_y=proj_params['cone_params']['pix_y']
    elif (proj_params['type'] == 'par'):
        pix_x = proj_params['pix_x']
        pix_y = proj_params['pix_y']
    else:
        pix_x = proj_params['pix_x']
        pix_y = proj_params['pix_y']

    if 'n_vox_x' not in rec_params.keys() or 'vol_col' not in rec_params.keys(): 
        rec_params['n_vox_z']=vol_params['n_vox_z']  
        rec_params['n_vox_x']=vol_params['n_vox_x'] 
        rec_params['n_vox_y']=vol_params['n_vox_y']

    if 'stop_thresh' not in rec_params.keys():
        rec_params['stop_thresh'] = DEFAULT_STOP_THRESH

    if 'verbose' not in rec_params.keys():
        rec_params['verbose']=False
        
    temp_vol_z = rec_params['n_vox_z']
    temp_vol_y = rec_params['n_vox_y']
    temp_vol_x = rec_params['n_vox_x']

    ROI_MASK = np.zeros((temp_vol_z,temp_vol_y,temp_vol_x),dtype=np.bool)
    y,x=np.ogrid[-temp_vol_y/2:temp_vol_y/2,-temp_vol_x/2:temp_vol_x/2]
    temp_roi_mask=create_circle_mask(y,x,np.array([0,0]),temp_vol_x/2)
    ROI_MASK[:]=temp_roi_mask
    rec_params['roi_mask']=np.reshape(ROI_MASK,temp_vol_z*temp_vol_y*temp_vol_x)

    if 'H_blur' in rec_params.keys():
        H = np.array(rec_params['H_blur'])
        rec,cost=mlCudaDebluropTomo(proj_data,weight_data,A,H,rec_params)
    else:
        rec,cost=mlopTomoPoisson(proj_data,weight_data,A,rec_params)
    
    if(rec_params['debug']):
        plt.plot(cost);plt.xlabel('Iter');plt.ylabel('Cost function');plt.show()

    return rec

