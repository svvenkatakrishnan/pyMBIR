#Reconsrtuct a experimental data set from the tomobank (https://tomobank.readthedocs.io/en/latest/) using the Gridrec algorithm and MBIR
#Requires downloading the relevant data set and having tomopy and dxchange installed for comparisons 
import numpy as np
import dxchange
import time
from pyMBIR.reconEngine import analytic,MBIR
import tomopy
from matplotlib import pyplot as plt 

fname = 'tomobank/tomo_00001_to_00006/tomo_00001/tomo_00001.h5'
disp_op=True
start = 1000 #start slice 
end = 1004 #end slice 
rot_center =  1024 
gpu_device = [0]
max_iter=150
MRF_P=1.2
MRF_SIGMA=0.35
stop_thresh=0.1 #percent 
view_subsamp=4 #sub-sampling ratio for sparse-view data 

#Read data
proj, flat, dark,theta = dxchange.read_aps_32id(fname, sino=(start, end))
proj=proj[::view_subsamp]
theta=theta[::view_subsamp]

proj_dims=np.array([proj.shape[1],proj.shape[0],proj.shape[2]])
print('Proj. dims : %s' % proj_dims)

miscalib={}
miscalib['delta_u']=proj_dims[2]/2-rot_center
miscalib['delta_v']=0
miscalib['phi']=0

rec_params={}
rec_params['gpu_index']=[0,1]
rec_params['num_iter']=max_iter
rec_params['MRF_P']=MRF_P
rec_params['MRF_SIGMA']=MRF_SIGMA
rec_params['debug']=False
rec_params['stop_thresh']=stop_thresh
rec_params['verbose']=True

proj_params={}
proj_params['type'] = 'par'
proj_params['dims']= proj_dims
proj_params['angles'] = theta
proj_params['alpha'] = np.array([np.pi/2])
proj_params['pix_x']= 1.0
proj_params['pix_y']= 1.0
proj_params['forward_model_idx']=2

vol_params={}
vol_params['vox_xy'] = 1.0
vol_params['vox_z'] = 1.0
vol_params['n_vox_x']=proj_dims[2]
vol_params['n_vox_y']=proj_dims[2]
vol_params['n_vox_z']=proj_dims[0]

#Start pipe-line
wt_data =  np.copy(proj).astype(np.float32)

proj = tomopy.normalize(proj, flat, dark)
proj = tomopy.minus_log(proj)

print('Starting Gridrec ..')
t=time.time()
rec_gr = tomopy.recon(proj, theta, center=rot_center, algorithm='gridrec')
elapsed_time_gr = time.time()-t
print('Time for reconstucting using TomoPy Gridrec: %f s' % (elapsed_time_gr))

proj = np.ascontiguousarray(proj.swapaxes(0,1)).astype(np.float32)
wt_data = np.ascontiguousarray(wt_data.swapaxes(0,1))

print('Starting  pyMBIR..')
t=time.time()
rec_pymbir = MBIR(proj,wt_data,proj_params,miscalib,vol_params,rec_params)
elapsed_time_pm = (time.time()-t)
print('Time for reconstucting using pyMBIR : %f s' % (elapsed_time_pm))

if(disp_op==True):
    slice_idx = proj_dims[0]//2
    scale_min = -1e-4
    scale_max = 3.5e-3
    plt.figure();plt.imshow(rec_gr[slice_idx],vmin=scale_min,vmax=scale_max,cmap='gray');plt.title('Gridrec')
    plt.figure();plt.imshow(rec_pymbir[slice_idx],vmin=scale_min,vmax=scale_max,cmap='gray');plt.title('MBIR')
    plt.show()
