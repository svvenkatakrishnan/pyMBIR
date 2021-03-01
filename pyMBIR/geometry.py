# Copyright (C) 2019, S.V.Venkatakrishnan <venkatakrisv@ornl.gov>
# All rights reserved. GPL v3 license. 
# This file is part of the pyMBIR package. Details of the copyright
# and user license can be found in the 'LICENSE' file distributed
# with the package.

#Functions to generate the A-matrix using spot operators and the ASTRA toolbox for different geometries 

import astra
import numpy as np

def createVecPar(det_x,det_y,angles,alpha,miscalib):
    """Function to generate the projection geometry vectors for parallel beam CT (including laminography)
    Inputs: det_x : Size of detector pixel along the x (column) direction
            det_y : Size of detector pixel along the y (row) direction
            angles: An array of angles in radians
            alpha : Laminographic angle [pi/2 radians implies standard parallel beam geometry ]
            miscalib : A dictionary containing miscalibration parameters 
                  'delta_u' : Center of rotation offset along the horizontal 
                  'delta_v' : Center of rotation offset along the vertical 
                  'phi' : Detector tilt angle in radians 
    Output : vec : A num_angles X 12 array 
    """
    vectors = np.zeros((len(angles)*len(alpha), 12))
    idx = 0
    for j in range(len(alpha)):
      for i in range(len(angles)):
        # ray direction
        vectors[idx,0] = np.sin(alpha[j])*np.sin(angles[i])
        vectors[idx,1] = -np.cos(angles[i])*np.sin(alpha[j])
        vectors[idx,2] = -np.cos(alpha[j])

        # center of detector
        vectors[idx,3] =miscalib['delta_u']*np.cos(angles[i])+miscalib['delta_v']*np.cos(alpha[j])*np.sin(angles[i]) 
        vectors[idx,4] = miscalib['delta_u']*np.sin(angles[i]) - miscalib['delta_v']*(np.cos(alpha[j])*np.cos(angles[i])) 
        vectors[idx,5] = miscalib['delta_v']*np.sin(alpha[j])
        
        # vector from detector pixel (0,0) to (0,1)
        vectors[idx,6] = np.cos(miscalib['phi'])*np.cos(angles[i])*det_x
        vectors[idx,7] = np.cos(miscalib['phi'])*np.sin(angles[i])*det_x
        vectors[idx,8] = np.sin(miscalib['phi'])*det_x

        # vector from detector pixel (0,0) to (1,0)
        vectors[idx,9] = (-np.sin(miscalib['phi'])*np.cos(angles[i])*np.cos(np.pi/2-alpha[j])+np.sin(angles[i])*np.sin(np.pi/2-alpha[j]))*det_y
        vectors[idx,10] = (-np.sin(miscalib['phi'])*np.sin(angles[i])*np.cos(np.pi/2-alpha[j])-np.sin(np.pi/2-alpha[j])*np.cos(angles[i]))*det_y
        vectors[idx,11] = np.cos(miscalib['phi'])*np.cos(np.pi/2-alpha[j])*det_y

        idx+=1
        
    return vectors

#Cone-beam CT with center offsets
def createVecCone(det_x,det_y,so,od,angles,miscalib):
    """Function to generate the projection geometry vectors for cone-beam CT
    Inputs: det_x : Size of detector pixel along the x (column) direction
            det_y : Size of detector pixel along the y (row) direction
            so    : Source-object distance 
            od    : Object-det distance 
            angles: An array of angles in radians
            miscalib : A dictionary containing miscalibration parameters 
                  'delta_u' : Center of rotation offset along the horizontal 
                  'delta_v' : Center of rotation offset along the vertical 
    Output : vec : A num_angles X 12 array 
    """
    vectors = np.zeros((len(angles), 12))
    idx = 0
    for i in range(len(angles)):
      #ray direction
      vectors[idx,0] = so*np.sin(angles[i])
      vectors[idx,1] = -so*np.cos(angles[i])
      vectors[idx,2] = 0
      
      #center of detector
      vectors[idx,3] = -od*np.sin(angles[i])+miscalib['delta_u']*np.cos(angles[i])+miscalib['delta_v']*np.sin(angles[i])
      vectors[idx,4] = od*np.cos(angles[i])+miscalib['delta_u']*np.sin(angles[i]) - miscalib['delta_v']*np.cos(angles[i])
      vectors[idx,5] = miscalib['delta_v']

      # vector from detector pixel (0,0) to (0,1)
      vectors[idx,6] = np.cos(angles[i])*det_x
      vectors[idx,7] = np.sin(angles[i])*det_x
      vectors[idx,8] = 0

      # vector from detector pixel (0,0) to (1,0)
      vectors[idx,9] = 0
      vectors[idx,10] = 0
      vectors[idx,11] = det_y
      idx+=1
        
    return vectors



def createVecParEuler(det_x,det_y,euler_angles,miscalib):
    """Function to generate the projection geometry vectors for single-particle cryo-EM like geometries 
    Inputs: det_x : Size of detector pixel along the x (column) direction
            det_y : Size of detector pixel along the y (row) direction
            euler_angles : An array of size num_angles X 3 of (alpha,beta,gamma) 
            miscalib : A dictionary containing miscalibration parameters 
                  'delta_u' : Center of rotation offset along the horizontal 
                  'delta_v' : Center of rotation offset along the vertical 
    Output : vec : A num_angles X 12 array 
    """
    num_angles=euler_angles.shape[0]
    vectors = np.zeros((num_angles, 12))
    idx=0
    alpha=euler_angles[:,0]
    beta=euler_angles[:,1]
    gamma=euler_angles[:,2]
    ca=np.cos(alpha)
    sa=np.sin(alpha)
    cb=np.cos(beta)
    sb=np.sin(beta)
    cg=np.cos(gamma)
    sg=np.sin(gamma)
    du=np.array(miscalib['delta_u'])
    dv=np.array(miscalib['delta_v'])
    for idx in range(num_angles):
        # ray direction, source to object center
        vectors[idx,0] = ca[idx]*cb[idx]*sg[idx]+sa[idx]*cg[idx] 
        vectors[idx,1] = sa[idx]*sg[idx]-ca[idx]*cb[idx]*cg[idx] 
        vectors[idx,2] = -ca[idx]*sb[idx] 
        
        # center of detector, object center to detector center 
        vectors[idx,3] = du[idx]*(-sa[idx]*cb[idx]*sg[idx]+ca[idx]*cg[idx]) + dv[idx]*(sb[idx]*sg[idx]) 
        vectors[idx,4] = du[idx]*(sa[idx]*cb[idx]*cg[idx]+ca[idx]*sg[idx]) + dv[idx]*(-sb[idx]*cg[idx])
        vectors[idx,5] =  du[idx]*(sa[idx]*sb[idx]) + dv[idx]*cb[idx] 

        # vector from detector pixel (0,0) to (0,1)
        vectors[idx,6] = det_x*(ca[idx]*cg[idx]-sa[idx]*cb[idx]*sg[idx]) 
        vectors[idx,7] = det_x*(ca[idx]*sg[idx]+sa[idx]*cb[idx]*cg[idx])
        vectors[idx,8] = det_x*(sa[idx]*sb[idx]) 

        # vector from detector pixel (0,0) to (1,0)
        vectors[idx,9] = det_y*(sb[idx]*sg[idx])
        vectors[idx,10] = det_y*(-sb[idx]*cg[idx])
        vectors[idx,11] = det_y*(cb[idx])

    return vectors

def createGeomParEuler(proj_params,miscalib,vol_params,gpu_idx):
    """Function to generate the projection and volume geometries for parallel beam tomography based on arbitary rotation angles specified by a euler angle pair of the covention y-z-y'
    Input :proj_params - Dictionary with keys 
                      - dims - 1 X 3 list of number of det rows, views and columns 
                      - pix_x - pixel size along rows 
                      - pix_y - pixel size along col. 
                      - euler_angles - num_views X 3 array of euler angles for each projection 
           miscalib - Miscaibration parameters; Dictionary with keys 
                  delta_u: center of rotation offset along columns 
                  delta_v: center of rotation offset along row dimension
           vol_params - Dictionary with keys 
                      - n_vox_x - Number of voxels along x direction 
                      - n_vox_y - Number of voxels along y direction 
                      - n_vox_z - Number of voxels along z direction 
                      - vox_xy - Voxel size in the x-y plane 
                      - vox_z - Voxel size along z (aligned with detector rows) 
           gpu_idx  - GPU index (python list)
    Output : proj_geom, vol_geom required by the astra tool-box
    """
    
    astra.astra.set_gpu_index(gpu_idx)
    
    det_row = proj_params['dims'][0]
    det_col = proj_params['dims'][2]

    pix_x = proj_params['pix_x']
    pix_y =  proj_params['pix_y']

    n_vox_x = vol_params['n_vox_x']
    n_vox_y = vol_params['n_vox_y']
    n_vox_z = vol_params['n_vox_z']
    vox_xy = vol_params['vox_xy']
    vox_z = vol_params['vox_z']
    minx=-n_vox_x*vox_xy/2
    maxx=n_vox_x*vox_xy/2
    miny=-n_vox_y*vox_xy/2
    maxy=n_vox_y*vox_xy/2
    minz=-n_vox_z*vox_z/2
    maxz=n_vox_z*vox_z/2
    vol_geom = astra.create_vol_geom(n_vox_y, n_vox_x, n_vox_z, minx, maxx, miny, maxy, minz, maxz)    
    vectors = createVecParEuler(pix_x,pix_y,proj_params['euler_angles'],miscalib)
    proj_geom = astra.create_proj_geom('parallel3d_vec',det_row,det_col,vectors)
    return proj_geom,vol_geom

def createGeomPar(proj_params,miscalib,vol_params,gpu_idx):
    """Function to generate the projection and volume geometries for standard parallel beam case (including lamingraphy) 
    Input :proj_params - Dictionary with keys 
                      - dims - 1 X 3 list of number of det rows, views and columns 
                      - pix_x - pixel size along rows 
                      - pix_y - pixel size along col. 
                      - angles - Array of angles in radians 
                      - alpha - Laminography angle in radians 
           miscalib - Miscaibration parameters; Dictionary with keys 
                  delta_u: center of rotation offset along columns 
                  delta_v: center of rotation offset along row dimension
                  phi : Detector tilt angle in radians 
           vol_params - Dictionary with keys 
                      - n_vox_x - Number of voxels along x direction 
                      - n_vox_y - Number of voxels along y direction 
                      - n_vox_z - Number of voxels along z direction 
                      - vox_xy - Voxel size in the x-y plane 
                      - vox_z - Voxel size along z (aligned with detector rows) 
           gpu_idx  - GPU index (python list)
    Output : proj_geom, vol_geom required by the astra tool-box
    """
    astra.astra.set_gpu_index(gpu_idx)
    
    det_row = proj_params['dims'][0]
    det_col = proj_params['dims'][2]

    pix_x = proj_params['pix_x']
    pix_y =  proj_params['pix_y']

    n_vox_x = vol_params['n_vox_x']
    n_vox_y = vol_params['n_vox_y']
    n_vox_z = vol_params['n_vox_z']
    vox_xy = vol_params['vox_xy']
    vox_z = vol_params['vox_z']
    minx=-n_vox_x*vox_xy/2
    maxx=n_vox_x*vox_xy/2
    miny=-n_vox_y*vox_xy/2
    maxy=n_vox_y*vox_xy/2
    minz=-n_vox_z*vox_z/2
    maxz=n_vox_z*vox_z/2
    vol_geom = astra.create_vol_geom(n_vox_y, n_vox_x, n_vox_z, minx, maxx, miny, maxy, minz, maxz)
    vectors = createVecPar(pix_x,pix_y,proj_params['angles'],proj_params['alpha'],miscalib) 
    proj_geom = astra.create_proj_geom('parallel3d_vec',det_row,det_col,vectors)
    return proj_geom,vol_geom

def createGeomCone(proj_params,miscalib,vol_params,gpu_idx):
    """Function to generate the projection and volume geometries for cone-beam CT
    Input :proj_params - Dictionary with keys 
                      - dims - 1 X 3 list of number of det rows, views and columns 
                      - angles - array of angles in radians 
                      - cone_params['pix_x'] - pixel size along rows 
                      - cone_params['pix_y'] - pixel size along col. 
                      - cone_params['src_orig'] - Source to sample distance 
                      - cone_params['orig_det'] - Sample to detector distance  
           miscalib - Miscaibration parameters; Dictionary with keys 
                  delta_u: center of rotation offset along columns 
                  delta_v: center of rotation offset along row dimension
           vol_params - Dictionary with keys 
                      - n_vox_x - Number of voxels along x direction 
                      - n_vox_y - Number of voxels along y direction 
                      - n_vox_z - Number of voxels along z direction 
                      - vox_xy - Voxel size in the x-y plane 
                      - vox_z - Voxel size along z (aligned with detector rows) 
           gpu_idx  - GPU index (python list)
    Output : proj_geom, vol_geom required by the astra tool-box
    """
    astra.astra.set_gpu_index(gpu_idx)
        
    det_row = proj_params['dims'][0]
    det_col = proj_params['dims'][2]

    pix_x = proj_params['cone_params']['pix_x']
    pix_y =  proj_params['cone_params']['pix_y']
    src_orig =  proj_params['cone_params']['src_orig']
    orig_det =  proj_params['cone_params']['orig_det']

    n_vox_x = vol_params['n_vox_x']
    n_vox_y = vol_params['n_vox_y']
    n_vox_z = vol_params['n_vox_z']
    vox_xy = vol_params['vox_xy']
    vox_z = vol_params['vox_z']
    minx=-n_vox_x*vox_xy/2
    maxx=n_vox_x*vox_xy/2
    miny=-n_vox_y*vox_xy/2
    maxy=n_vox_y*vox_xy/2
    minz=-n_vox_z*vox_z/2
    maxz=n_vox_z*vox_z/2
    vol_geom = astra.create_vol_geom(n_vox_y, n_vox_x, n_vox_z, minx, maxx, miny, maxy, minz, maxz)
    if('vecs' in miscalib):
        vectors = miscalib['vecs']
    else:
        vectors = createVecCone(pix_x,pix_y,src_orig,orig_det,proj_params['angles'],miscalib)
    proj_geom = astra.create_proj_geom('cone_vec', det_row, det_col, vectors)
    
    return proj_geom,vol_geom

def generateAmatrix(proj_params,miscalib,vol_params,gpu_idx):
    """
    Generate the A matrix using spot operators
    Inputs: proj_params: Dictionary of the projection related parameters 
            miscalib: Dictionary of miscalibration paramaters such as center of offset rotation and/or tilt 
            vol_params: Dictionary of reconstruction volume parameters 
            gpu_idx: List of gpus to use 
    """
    if(proj_params['type'] == 'par'):
        proj_geom,vol_geom = createGeomPar(proj_params,miscalib,vol_params,gpu_idx)
    elif (proj_params['type'] == 'cone'):
        proj_geom,vol_geom = createGeomCone(proj_params,miscalib,vol_params,gpu_idx)
    elif (proj_params['type'] == 'par_euler'):
        proj_geom,vol_geom = createGeomParEuler(proj_params,miscalib,vol_params,gpu_idx)
    else:
        print('Unrecognized type for projector')
        
    proj_id = astra.create_projector('cuda3d', proj_geom, vol_geom)   
    #opTomo based Projection function 
    A = astra.OpTomo(proj_id)
    return A


