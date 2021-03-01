# -----------------------------------------------------------------------
# Created by S.V. Venkatakrishnan
# Code to simulate laminography data and reconstruct it 
# -----------------------------------------------------------------------

import numpy as np
import time
from pyMBIR.utils import createTransmission,rmse
from pyMBIR.geometry import generateAmatrix
from pyMBIR.reconEngine import MBIR
from generate_phantom import cube_phantom
from matplotlib import pyplot as plt 

#Detector and experiment parameters
num_angles = 128
det_row = 256
det_col = 256
#Input flux and noise 
I0 = 2e3
noise_std = 1
off_center_u = -15.75 #Center of rotation offset in units of pixels 
off_center_v=0
det_tilt=0 
lam_angle = 70 #Laminography angle in degrees

#Cube phantom parameters
cube_z=50
cube_y=180
cube_x=180
density=0.01

det_x=1.0
det_y=1.0
vox_xy=1.0
vox_z=1.0

# Geometry for laminography 
alpha=np.array([lam_angle])

#Miscalibrations - detector offset, tilt
miscalib={}
miscalib['delta_u']=off_center_u*det_x
miscalib['delta_v']=off_center_v*det_y
miscalib['phi']=det_tilt*np.pi/180

#MRF parameters
MRF_P = 1.2 #1.2
MRF_SIGMA = 0.5 
NUM_ITER = 200
gpu_index = 0

disp_op = True

######End of inputs#######

im_size = np.int(det_col*det_x/vox_xy) # n X n X n_slice volume             
num_slice = np.int(det_row*det_y/vox_z)
obj = cube_phantom(num_slice,im_size,cube_z,cube_y,cube_x,density)

# ----------------------------------
# Parallel beam Laminography Angle
# ----------------------------------
alpha=alpha*np.pi/180

# Parameters: width of detector column, height of detector row, #rows, #columns
angles = np.linspace(0, 2*np.pi, num_angles, False)

proj_dims=np.array([det_row,num_angles,det_col])

proj_params={}
proj_params['type'] = 'par'
proj_params['dims']= proj_dims
proj_params['angles'] = angles
proj_params['alpha'] = np.array([alpha])
proj_params['forward_model_idx']=2

proj_params['pix_x']= det_x
proj_params['pix_y']= det_y

vol_params={}
vol_params['vox_xy'] = vox_xy
vol_params['vox_z'] = vox_z
vol_params['n_vox_x']=det_col
vol_params['n_vox_y']=det_col
vol_params['n_vox_z']=det_row

A=generateAmatrix(proj_params,miscalib,vol_params,gpu_index)
proj_data = A*obj
proj_data=proj_data.astype(np.float32).reshape(det_row,num_angles,det_col)

#Simulate Poisson like statistics using Gaussian approximation
weight_data = createTransmission(proj_data,I0,noise_std)

#Test projector
print('Min/Max %f/%f of weight data' %(weight_data.min(),weight_data.max()))
proj_data = np.log(I0/weight_data)

#Display object
print('Actual projection shape (%d,%d,%d)'% proj_data.shape)
plt.imshow(proj_data.swapaxes(0,1)[0],cmap='gray');plt.title('Projection image');plt.show()

rec_params={}
rec_params['num_iter'] = NUM_ITER
rec_params['gpu_index']= gpu_index
rec_params['MRF_P'] = MRF_P
rec_params['MRF_SIGMA'] = MRF_SIGMA
rec_params['huber_T']=5
rec_params['huber_delta']=0.1
rec_params['sigma']=1
rec_params['reject_frac']=.1
rec_params['verbose']=True
rec_params['debug']=False
rec_params['stop_thresh']=0.001

recon_mbir=MBIR(proj_data,weight_data,proj_params,miscalib,vol_params,rec_params)

rmse_mbir = rmse(recon_mbir,obj)
print('MBIR RMSE :%f' %(rmse_mbir))

if disp_op == True:
    slice_idx = num_slice//2
    plt.figure();plt.imshow(obj[slice_idx],cmap='gray');plt.title('Ground truth')
    plt.figure();plt.imshow(recon_mbir[slice_idx],cmap='gray');plt.title('MBIR')
    plt.show()


