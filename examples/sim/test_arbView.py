# -----------------------------------------------------------------------
# Created by S.V. Venkatakrishnan
# Code to simulate single particle CryoEM like data and reconstruct
# MBIR is formulated as: 0.5*||y-HAx||_{W}^{2} + R(x)
# -----------------------------------------------------------------------
import numpy as np
import time
from pyMBIR.utils import rmse,genCTF
from pyMBIR.geometry import generateAmatrix
from pyMBIR.forwardmodel import forwardProject
from pyMBIR.reconEngine import MBIR
from generate_phantom import cube_phantom
from matplotlib import pyplot as plt 

#Detector and experiment parameters
det_row=200
det_col=200 
num_angles=220 #Simulated number of angles

#Cube phantom parameters
density=0.1 #Max value of test object used in simulation 
cube_z=100
cube_y=20
cube_x=60

#Input flux and noise 
noise_lev=0.1 #Max amplitude of the noise relative to the signal; 0 - no noise, 1.0 -> 0 dB PSNR 

#Detector pixel size 
det_x=1.0 #arbitrary units 
det_y=1.0

#Recon voxel size 
vox_xy=1.0
vox_z=1.0

#Parameters of the contrast transfer function based on the equation
# exp{-att_coeff * |f|} * sin(-\pi*c1*f^2 + (\pi/2)*c2*f^4)
c1 = 10.0 #delta_z*lambda
c2 = 1.0 #Cs*lambda^3
att_coeff = 1.0

#Recon: MRF parameters
MRF_P = 1.2
MRF_SIGMA = 0.3 
NUM_ITER = 250
gpu_index = [0] #GPU index
disp_op = True 

######End of inputs#######

#Create simulated data 
euler_angles=(2*np.pi)*np.random.rand(num_angles,3)

im_size = np.int(det_col*det_x/vox_xy) # n X n X n_slice volume             
num_slice = np.int(det_row*det_y/vox_z)
obj = cube_phantom(num_slice,im_size,cube_z,cube_y,cube_x,density)

# Parameters: width of detector column, height of detector row, #rows, #columns
proj_dims=np.array([det_row,num_angles,det_col])

proj_params={}
proj_params['type'] = 'par_euler' #parallel beam geometry with Euler angles specified 
proj_params['dims']= proj_dims
proj_params['euler_angles'] = euler_angles
proj_params['forward_model_idx']=4
proj_params['pix_x']= det_x
proj_params['pix_y']= det_y

vol_params={}
vol_params['vox_xy']=vox_xy
vol_params['vox_z']=vox_z
vol_params['n_vox_x']=det_col
vol_params['n_vox_y']=det_col
vol_params['n_vox_z']=det_row

#Simulated shifts of the projection data 
miscalib={}
miscalib['delta_u']=(det_col/20)*np.random.randn(len(euler_angles))
miscalib['delta_v']=(det_row/20)*np.random.randn(len(euler_angles))

#Contrast transfer function for each projection 
H = np.zeros((num_angles,det_row,det_col)).astype(np.csingle)
for idx in range(num_angles):
    H[idx]=genCTF(c1+(.05*c1)*np.random.randn(1),c2+(.05*c2)*np.random.randn(1),att_coeff+(.05*att_coeff)*np.random.randn(1),det_row,det_col)

#Generate A matrix based on the geometry of the problem 
A=generateAmatrix(proj_params,miscalib,vol_params,gpu_index)

#Create projection data y = H*A*x + n 
y = forwardProject(A,H,obj,[det_row,num_angles,det_col])
proj_data= y + (noise_lev*y.max())*np.random.randn(len(y))

#Prepare projection data for reconstruction 
proj_data=proj_data.astype(np.float32).reshape(det_row,num_angles,det_col)
#Display object
print('Actual projection shape (%d,%d,%d)'% proj_data.shape)
plt.imshow(proj_data.swapaxes(0,1)[0],cmap='gray');plt.title('Projection image');plt.show()

weight_data=np.ones_like(proj_data)
weight_data=weight_data


####################Reconstruction#########################3

rec_params={}
rec_params['num_iter'] = NUM_ITER
rec_params['gpu_index']= gpu_index
rec_params['MRF_P'] = MRF_P
rec_params['MRF_SIGMA'] = MRF_SIGMA
rec_params['debug']=False
rec_params['sigma']=1
rec_params['reject_frac']=.01
rec_params['stop_thresh']=.0001
rec_params['verbose']=True
rec_params['H_blur']=H

t=time.time()
recon_mbir=MBIR(proj_data,weight_data,proj_params,miscalib,vol_params,rec_params)
elapsed_mbir=time.time()-t
print('Time taken for MBIR-Cryo Recon : %f s' % (elapsed_mbir))

rmse_mbir = rmse(recon_mbir,obj)
print('MBIR RMSE :%f' %(rmse_mbir))

if disp_op == True:
    slice_idx = num_slice//2
    plt.figure();plt.imshow(obj[slice_idx],cmap='gray');plt.title('Ground truth')
    plt.figure();plt.imshow(recon_mbir[slice_idx],cmap='gray');plt.title('MBIR')
    plt.show()
