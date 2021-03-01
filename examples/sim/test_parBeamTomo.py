# -----------------------------------------------------------------------
# Created by S.V. Venkatakrishnan
# Code to simulate parallel beam data and reconstruct it 
# -----------------------------------------------------------------------

import numpy as np
import time
from pyMBIR.utils import createTransmission,rmse
from pyMBIR.geometry import generateAmatrix
from pyMBIR.reconEngine import analytic,MBIR
from generate_phantom import cube_phantom
from matplotlib import pyplot as plt 

#Detector and experiment parameters
num_angles = 100
det_row = 256
det_col = 256

#Input flux and noise 
I0 = 2e3
noise_std = 1
off_center_u = 18.5 #Center of rotation offset from center of detector in units of pixels 
off_center_v=0
det_tilt=0 
det_x=1.0
det_y=1.0

#Cube phantom parameters
cube_z=100
cube_y=40
cube_x=60
density=0.01

vox_xy=1.0
vox_z=1.0
view_subsamp=1 #Sub-set the views by this factor
#MRF parameters
MRF_P = 1.2 
MRF_SIGMA = 0.3 
NUM_ITER = 150
gpu_index = [0]
#FBP cut off
f_c = 0.5

disp_op = True
######End of inputs#######



#Miscalibrations - detector offset, tilt
miscalib={}
miscalib['delta_u']=off_center_u*det_x
miscalib['delta_v']=off_center_v*det_y
miscalib['phi']=det_tilt*np.pi/180

lam_angle = 90 #Laminography angle; 90 => conventional parallel beam CT
alpha=np.array([lam_angle])*np.pi/180
num_angles=num_angles//view_subsamp

im_size = np.int(det_col*det_x/vox_xy) # n X n X n_slice volume             
num_slice = np.int(det_row*det_y/vox_z)
obj = cube_phantom(num_slice,im_size,cube_z,cube_y,cube_x,density)

# ----------------------------------
# Parallel beam Laminography Vectors
# ----------------------------------
# Parameters: width of detector column, height of detector row, #rows, #columns
angles = np.linspace(0, np.pi, num_angles, False)
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


print(proj_data.shape)

rec_params={}
rec_params['num_iter'] = NUM_ITER
rec_params['gpu_index']= gpu_index
rec_params['MRF_P'] = MRF_P
rec_params['MRF_SIGMA'] = MRF_SIGMA
rec_params['debug']=False
rec_params['verbose']=True
rec_params['stop_thresh']=0.01

#params for fbp
rec_params['filt_type']='Ram-Lak'
rec_params['filt_cutoff']=f_c

recon_fbp=np.float32(analytic(proj_data,proj_params,miscalib,vol_params,rec_params))
for idx in range(recon_fbp.shape[0]):
    recon_fbp[idx] = np.flipud(recon_fbp[idx])

recon_mbir=np.float32(MBIR(proj_data,weight_data,proj_params,miscalib,vol_params,rec_params))

rmse_fbp = rmse(recon_fbp,obj)
rmse_mbir = rmse(recon_mbir,obj)
print('FBP RMSE :%f' %(rmse_fbp))
print('MBIR RMSE :%f' %(rmse_mbir))

if disp_op == True:
    slice_idx = num_slice//2
    plt.figure();plt.imshow(obj[slice_idx],cmap='gray');plt.title('Ground truth')
    plt.figure();plt.imshow(recon_fbp[slice_idx],cmap='gray');plt.title('FBP')
    plt.figure();plt.imshow(recon_mbir[slice_idx],cmap='gray');plt.title('MBIR')
    plt.show()
