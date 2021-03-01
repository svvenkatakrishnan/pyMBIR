#Test cone-beam reconstruction algorithm by simulating and reconstructing a data set 

import numpy as np
import time
from pyMBIR.reconEngine import analytic,MBIR
from pyMBIR.geometry import generateAmatrix
from generate_phantom import cube_phantom
from matplotlib import pyplot as plt 

#Detection parameters 
det_row = 256
det_col = 256
num_angles = 64

#Cube phantom parameters
density=1.0 #Max value of test object used in simulation 
cube_z=100
cube_y=40
cube_x=60

I0=2000 #Input flux 
noise_lev=1.0

det_pix_x=30e-3 #mm
det_pix_y=30e-3 #mm
src_orig=20 #mm
orig_det=200 #mm
vox_xy=5e-3 #mm
vox_z=5e-3
n_vox_x=256
n_vox_y=256
n_vox_z=256

center_offset_col=42.5 #Center of rotation in units of pixels away from the center of the detector  
center_offset_row=0

gpu_index = [0]

#Recon parameters
MRF_P = 1.2
MRF_SIGMA = 0.001
max_iter = 150 
disp_op=True

proj_dims=np.array([det_row,num_angles,det_col])

angles = np.linspace(0,2*np.pi,num_angles, False)


proj_params={}
proj_params['type'] = 'cone'
proj_params['dims']= proj_dims
proj_params['angles'] = angles
cone_params={}
cone_params['pix_x'] = det_pix_x
cone_params['pix_y'] = det_pix_y 
cone_params['src_orig'] = src_orig
cone_params['orig_det'] = orig_det 
proj_params['cone_params']=cone_params
proj_params['forward_model_idx']=2

vol_params={}
vol_params['vox_xy'] = vox_xy
vol_params['vox_z'] = vox_z
vol_params['n_vox_x']=n_vox_x
vol_params['n_vox_y']=n_vox_y
vol_params['n_vox_z']=n_vox_z

miscalib={}
miscalib['delta_u']=center_offset_col*det_pix_x
miscalib['delta_v']=center_offset_row*det_pix_y

A=generateAmatrix(proj_params,miscalib,vol_params,gpu_index)

#Generate data with noise 
x = cube_phantom(n_vox_z,n_vox_y,cube_z,cube_y,cube_x,density) 
proj = A*x

weight_data=I0*np.exp(-proj)
weight_data=weight_data+np.sqrt(noise_lev*weight_data)*np.random.randn(*weight_data.shape)

proj_data=np.log(I0/weight_data)

print('Min/Max value of projection %f,%f' %(weight_data.min(),weight_data.max()))

#Reshape arrays so it is of the form rows, angles, columns 
proj_data = proj_data.reshape(det_row,num_angles,det_col) 
weight_data=weight_data.reshape(det_row,num_angles,det_col)

plt.imshow(proj_data.swapaxes(0,1)[0],cmap='gray');plt.title('Projection image');plt.show()

#Reconstruction
rec_params={}
rec_params['gpu_index']=gpu_index
rec_params['num_iter']=max_iter
rec_params['MRF_P']=MRF_P
rec_params['MRF_SIGMA']= MRF_SIGMA*(1.0/vox_xy**2) 
rec_params['debug']=False
rec_params['verbose']=True
rec_params['stop_thresh']=0.001 #percentage 

rec_fdk=analytic(proj_data,proj_params,miscalib,vol_params,rec_params)

print('Starting MBIR')
print('Regularization parameter = %f' % rec_params['MRF_SIGMA'])
t1=time.time()
rec_mbir=MBIR(proj_data,weight_data,proj_params,miscalib,vol_params,rec_params)
t2=time.time()-t1
print('Elapsed time %f s' % (t2))

if disp_op == True:
    slice_idx = n_vox_z//2
    plt.figure();plt.imshow(x[slice_idx],cmap='gray');plt.title('Ground truth')
    plt.figure();plt.imshow(rec_fdk[slice_idx],cmap='gray');plt.title('FDK')
    plt.figure();plt.imshow(rec_mbir[slice_idx],cmap='gray');plt.title('MBIR')
    plt.show()
