#Reconstruct a experimental data set from Henri Der Sarkissian et al. (Der Sarkissian, Henri, et al. "A Cone-Beam X-Ray CT Data Collection designed for Machine Learning." arXiv preprint arXiv:1905.04787 (2019).) using the FDK and MBIR
#Requires downloading the relevant data set 
import numpy as np
from dxchange.reader import read_tiff_stack,read_tiff 
import time
from pyMBIR.reconEngine import analytic,MBIR
import os
from matplotlib import pyplot as plt 

data_path = 'Walnut8/Projections/tubeV2'
proj_str ='scan_000280.tif'
dark_img = 'di000000.tif'
brt_img = 'io000000.tif'
vecs_name = 'scan_geom_corrected.geom' #Corrected vectors as per the astra toolbox for cone beam CT  

num_angles = 1200
ang_step = 0.3 #degrees
src_orig = 66 #mm
orig_det = 124 #mm
det_pix = 0.1496 #mm
n_vox_x = 380
n_vox_y = 380
n_vox_z = 450
opt_mag = 3.016
det_row = 972
det_col = 768 
recon_voxel_fact = 0.23 #Nearly 100 micron side 
view_subsamp = 4
disp_op=True

det_pix_x = opt_mag*det_pix
det_pix_y = det_pix_x
print('Effective Detector Pixel size %f mm' %(det_pix_x))
det_size=np.array([det_row,det_col])
vol_xy = det_pix_x*recon_voxel_fact
vol_z = vol_xy

#MBIR params
gpu_index = [0,1]
NUM_ITER = 150
MRF_P = 1.2
MRF_SIGMA = 0.05

vecs = np.loadtxt(os.path.join(data_path, vecs_name))
vecs = vecs[range(0,num_angles)][::view_subsamp]

raw_data = (read_tiff_stack(os.path.join(data_path,proj_str),range(0,num_angles)))[::view_subsamp]
drk_img  = read_tiff(os.path.join(data_path,dark_img))
brt_img  = np.mean(read_tiff_stack(os.path.join(data_path,brt_img),range(0,2)),axis=0)

num_angles/=view_subsamp

weight_data = raw_data - drk_img
proj_data = np.log((brt_img-drk_img)/weight_data)
angles = np.linspace(0,2*np.pi,num_angles)

weight_data=weight_data.swapaxes(1,2).swapaxes(0,1).astype(np.float32)
proj_data=proj_data.swapaxes(1,2).swapaxes(0,1).astype(np.float32)

print('Actual projection shape (%d,%d,%d)'% proj_data.shape)

################Recon###############

proj_dims=np.array([det_row,num_angles,det_col]).astype(np.uint16)
proj_params={}
proj_params['type']='cone'
proj_params['dims']=proj_dims
proj_params['angles']=angles
cone_params={}
cone_params['pix_x'] = det_pix_x
cone_params['pix_y'] = det_pix_y 
cone_params['src_orig'] = src_orig
cone_params['orig_det'] = orig_det
proj_params['cone_params']=cone_params
proj_params['forward_model_idx']=2

miscalib={}
miscalib['vecs']=vecs

rec_params={}
rec_params['num_iter'] = NUM_ITER
rec_params['gpu_index']= gpu_index
rec_params['MRF_P']=MRF_P 
rec_params['MRF_SIGMA']= MRF_SIGMA*(1.0/vol_xy**2)
print('Regularization parameter = %f' % rec_params['MRF_SIGMA'])
rec_params['debug']=False
rec_params['sigma']=1
rec_params['reject_frac']=0.1
rec_params['verbose']=True
rec_params['stop_thresh'] = 0.001 #percent
rec_params['vol_row']=n_vox_z
rec_params['vol_col']=n_vox_x

vol_params={}
vol_params['vox_xy']=vol_xy
vol_params['vox_z']=vol_z
vol_params['n_vox_x']=n_vox_x
vol_params['n_vox_y']=n_vox_y
vol_params['n_vox_z']=n_vox_z

rec_fdk=analytic(proj_data,proj_params,miscalib,vol_params,rec_params)
rec_mbir = MBIR(proj_data,weight_data,proj_params,miscalib,vol_params,rec_params)

if(disp_op==True):
    slice_idx = n_vox_z//2
    plt.figure();plt.imshow(rec_fdk[slice_idx],cmap='gray');plt.title('FDK')
    plt.figure();plt.imshow(rec_mbir[slice_idx],cmap='gray');plt.title('MBIR')
    plt.show()





