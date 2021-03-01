# Copyright (C) 2019, S.V.Venkatakrishnan <venkatakrisv@ornl.gov>
# All rights reserved. GPL v3 License.
# This file is part of the pyMBIR package. Details of the copyright
# and user license can be found in the 'LICENSE' file distributed
# with the package.

#Core reconstruction routines. See geometry.py or the examples folder for details of the input parameters for the different projection geometries 

from pyMBIR.optimization import *
from pyMBIR.geometry import * 
from pyMBIR.forwardmodel import *
from pyMBIR.prior_model.mrf_prior import qGGMRFfuncs,ncg_qGGMRF_funcs
from pyMBIR.utils import stoppingCritVol,create_circle_mask
import gc
import time

def fdkCuda(proj_data,proj_params,miscalib,vol_params,rec_params):
    """Function for simple GPU based FDK reconstruction 
    Inputs: proj_data : A num_rows X num_angles X num_columns array 
            proj_params : Dictionary of parameter associated with the projection data 
            miscalib : Dictionary of parameters associated with miscalibrations (center of rotation, tilt)
            vol_params : Dictionary of parameters associated with the reconstuction volume
            rec_params: Dictionary of parameters associated with the reconstruction algorithm 
    Output : recon : A num_rows X num_cols X num_cols array  
    """
    proj_geom,vol_geom=createGeomCone(proj_params,miscalib,vol_params,rec_params['gpu_index'])
    proj_data_id = astra.data3d.link('-sino', proj_geom, proj_data)
    # Create a data object for the reconstruction
    rec_id = astra.data3d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FDK_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_data_id
    alg_id = astra.algorithm.create(cfg)
    if(rec_params['verbose']):
        print('Starting FDK ..')
        t=time.time()    
    astra.algorithm.run(alg_id)

    if(rec_params['verbose']):
        elapsed_time = time.time()-t
        print('Time taken = %f'%(elapsed_time))
    
    # Get the result
    recon = astra.data3d.get(rec_id)
    
    #Clean up 
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(proj_data_id)
    
    return recon 

def backProjCuda(proj_data,proj_params,miscalib,vol_params,rec_params):
    """Function for simple GPU based back-projection reconstruction 
    Inputs: proj_data : A num_rows X num_angles X num_columns array 
            proj_params : Dictionary of parameter associated with the projection data 
            miscalib : Dictionary of parameters associated with miscalibrations (center of rotation etc.)
            vol_params : Dictionary of parameters associated with the reconstuction volume
            rec_params: Dictionary of parameters associated with the reconstruction algorithm 
    Output : recon : A num_rows X num_cols X num_cols array  
    """
    proj_geom,vol_geom = createGeomPar(proj_params,miscalib,vol_params,rec_params['gpu_index'])
    proj_data_id = astra.data3d.link('-sino', proj_geom, proj_data)
    # Create a data object for the reconstruction
    rec_id = astra.data3d.create('-vol', vol_geom)
    cfg = astra.astra_dict('BP3D_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_data_id 
    alg_id = astra.algorithm.create(cfg)

    if(rec_params['verbose']==True):
        print('Starting Back-Projection ..')
        t=time.time()    
    astra.algorithm.run(alg_id)

    if(rec_params['verbose']==True):
        elapsed_time = time.time()-t
        print('Time taken = %f'%(elapsed_time))
    
    # Get the result
    recon = astra.data3d.get(rec_id)
    
    #Clean up 
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(proj_data_id)
    
    return recon

def fbpCuda(proj_data,proj_params,miscalib,vol_params,rec_params):
    """Function for simple GPU based filtered back-projection reconstruction 
    Inputs: proj_data : A num_rows X num_angles X num_columns array 
            proj_params : Dictionary of parameter associated with the projection data 
            miscalib : Dictionary of parameters associated with miscalibrations (center of rotation etc.)
            vol_params : Dictionary of parameters associated with the reconstuction volume
            rec_params: Dictionary of parameters associated with the reconstruction algorithm 
    Output : recon : A num_rows X num_cols X num_cols array  
    """
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
    recon = np.zeros((n_vox_z,n_vox_x,n_vox_y))
    vol_geom = astra.create_vol_geom(n_vox_y, n_vox_x, n_vox_z, minx, maxx, miny, maxy, minz, maxz)    
    proj_geom = astra.create_proj_geom('parallel', pix_x, det_col,proj_params['angles'])
    proj_geom= astra.geom_postalignment(proj_geom,miscalib['delta_u']) 
    
    rec_id = astra.data2d.create('-vol', vol_geom)

    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['FilterType'] = rec_params['filt_type']
    cfg['FilterD']=rec_params['filt_cutoff']    

    for idx in range(n_vox_z):
        proj_data_id = astra.data2d.link('-sino', proj_geom,proj_data[idx])
        cfg['ProjectionDataId'] = proj_data_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        # Get the result
        recon[idx] = astra.data2d.get(rec_id)
        #Clean up 
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(proj_data_id)

    astra.data2d.delete(rec_id)    
    return recon 

def sirtCudaopTomo(proj_data,A,rec_params):
    """Function for GPU based SIRT based on the opTomo function. This uses less GPU memory but moves large arrays to adn from GPU (sub-optimal) 
    Inputs: proj_data : A num_rows X num_angles X num_columns array 
            A : Spot operator based forward projection matrix 
            rec_params: Dictionary of parameters associated with the reconstruction algorithm 
    Output : recon : A num_rows X num_cols X num_cols array  
    """
    
    #Generate the geometry 
    det_row = proj_data.shape[0]
    num_views = proj_data.shape[1]
    det_col = proj_data.shape[2]

    vol_z = rec_params['n_vox_z']
    vol_x = rec_params['n_vox_x']
    vol_y = rec_params['n_vox_y']
    
    #Array to save recon
    if 'x_init' in rec_params.keys():
      recon = rec_params['x_init'].reshape(vol_z*vol_x*vol_y)
    else:
      recon = np.zeros(vol_z*vol_x*vol_y,dtype=np.float32)

    #Pre-compute diagonal scaling matrices ; one the same size as the image    and the other the same as data 
    x_ones=np.ones((vol_z,vol_y,vol_x),dtype=np.float32)
    temp_proj= A*x_ones
    R = 1/temp_proj
    R[np.isinf(R)]=0
    R=np.array(R,dtype=np.float32)

    #Initialize a sinogram of all ones
    y_ones=np.ones(det_row*det_col*num_views)
    temp_backproj=A.T*y_ones
    C = 1/temp_backproj
    C[np.isinf(C)]=0
    C=np.array(C,dtype=np.float32)

    #Clear all memory of un-used variables 
    #data #x_ones #y_ones
    del x_ones,temp_proj,y_ones,temp_backproj

    #Flatten projection data to a vector 
    proj_data = proj_data.reshape(det_row*det_col*num_views).astype(np.float32)
    
    if(rec_params['verbose']):
      print('Starting %d SIRT iterations ..' % (rec_params['num_iter']))
      t=time.time()
    
    #SIRT loop x_k+1 = x_k + C*AT*R*(y-Ax_k)
    for iter_num in range(rec_params['num_iter']):
      if(rec_params['verbose']):
        print('Iter number %d of %d in %f sec' %(iter_num,rec_params['num_iter'],time.time()-t))
      temp = R*(proj_data-(A*recon))
      temp2 = A.T*temp
      recon += C*temp2
      del temp,temp2 #Release memory 

    if(rec_params['verbose']):
      elapsed_time = time.time()-t
      print('Time for %d iterations = %f'%(rec_params['num_iter'],elapsed_time))

    recon= recon.reshape(vol_z,vol_y,vol_x)
    return recon

def mbiropTomo(proj_data,A,rec_params):
    """Function for MBIR based on the opTomo function. This uses less GPU memory but moves large arrays to and from GPU (sub-optimal) 
    Inputs: proj_data : A num_rows X num_angles X num_columns array 
            A : Spot operator based forward projection matrix 
            rec_params: Dictionary of parameters associated with the reconstruction algorithm 
    Output : recon : A num_rows X num_cols X num_cols array  
    """

    DEFAULT_STOP_THRESH = 1
    MIN_ITER = 5

    det_row = proj_data.shape[0]
    num_views = proj_data.shape[1]
    det_col = proj_data.shape[2]

    vol_z = rec_params['n_vox_z']
    vol_x = rec_params['n_vox_x']
    vol_y = rec_params['n_vox_y']
    
    #Prior model initializations
    mrf_cost,grad_prior,hessian_prior=qGGMRFfuncs()

    vol_size = vol_z*vol_y*vol_x
    proj_size = det_row*det_col*num_views
    
    #Array to save recon
    if 'x_init' in rec_params.keys():
      x_recon = rec_params['x_init'].reshape(vol_size)
      z_recon = rec_params['x_init'].reshape(vol_size)
    else:     
      x_recon = np.ascontiguousarray(np.zeros(vol_size,dtype=np.float32))
      z_recon = np.ascontiguousarray(np.zeros(vol_size,dtype=np.float32))

    #Flatten projection data to a vector 
    proj_data = proj_data.reshape(proj_size).astype(np.float32)
    
    if(rec_params['verbose']):
      print('Starting %d MBIR iterations ..' % (rec_params['num_iter']))

    #Compute Lipschitz of gradient
    temp_backproj=LipschitzForward(vol_size,A,np.ones(*proj_data.shape))
    x_ones = np.ones(vol_size,dtype=np.float32)
    hessian_prior(x_ones,temp_backproj,vol_z,vol_y,vol_x,rec_params['MRF_SIGMA'])
    L = temp_backproj.max()

    if(rec_params['verbose']):
      print('Lipschitz constant = %f' %(L))

    del x_ones,temp_backproj
    
    #Initialize variables for Nesterov method
    #ASSUME both x and z are set to zero 
    t_nes = 1
    t=time.time()
    gradient = np.ascontiguousarray(np.zeros(vol_size,dtype=np.float32))
    temp_cost = np.zeros(1,dtype=np.float32)
    cost = np.zeros(rec_params['num_iter'])
    #MBIR loop x_k+1 = x_k + func(gradient)
    for iter_num in range(rec_params['num_iter']):
      if(rec_params['verbose']):
        print('Iter number %d of %d in %f sec' %(iter_num,rec_params['num_iter'],time.time()-t))
      error = (A*x_recon)-proj_data
      gradient = A.T*error
      #Cost compute for Debugging
      if rec_params['debug'] == True:
          temp_cost_forward=0.5*(error*error).sum()
          mrf_cost(x_recon,temp_cost,vol_z,vol_y,vol_x,rec_params['MRF_P'],rec_params['MRF_SIGMA'])
          cost[iter_num]=temp_cost_forward+temp_cost
          print('Forward Cost %f, Prior Cost %f' %(temp_cost_forward,temp_cost[0]))
          temp_cost = temp_cost*0
          if(iter_num > 0 and (cost[iter_num]-cost[iter_num-1])>0):
              print('Cost went up!')
              t_nes=1 #reset momentum 
      grad_prior(x_recon,gradient,vol_z,vol_y,vol_x,rec_params['MRF_P'],rec_params['MRF_SIGMA']) #accumulates gradient from prior
      x_prev = x_recon
      x_recon,z_recon,t_nes=nesterovOGM2update(x_recon,z_recon,t_nes,gradient,L)
      if iter_num>MIN_ITER and stoppingCritVol(x_recon,x_prev,rec_params['stop_thresh'],rec_params['roi_mask']):
              break 
      gc.collect() #the call to the C-code grad_prior seems to cause memory to grow; this is a basic fix. TODO: Better memory fix
      
    elapsed_time = time.time()-t
    if(rec_params['verbose']):
      print('Time for %d iterations = %f'%(rec_params['num_iter'],elapsed_time))

    recon= x_recon.reshape(vol_z,vol_y,vol_x)
    return recon,cost

def mbiropTomoPoisson(proj_data,weight_data,A,rec_params):
    """Function for MBIR based on the opTomo function and a quadratic approximation to the log-likelihood term. This uses less GPU memory but moves large arrays to and from GPU (sub-optimal) 
    Inputs: proj_data : A num_rows X num_angles X num_columns array 
            weight_data : A num_rows X num_angles X num_columns array containting the noise variance values 
            A : Spot operator based forward projection matrix 
            rec_params: Dictionary of parameters associated with the reconstruction algorithm 
    Output : recon : A num_rows X num_cols X num_cols array  
    """    
    MIN_ITER = 5
    det_row = proj_data.shape[0]
    num_views = proj_data.shape[1]
    det_col = proj_data.shape[2]

    vol_z = rec_params['n_vox_z']
    vol_x = rec_params['n_vox_x']
    vol_y = rec_params['n_vox_y']
 
    #Prior model initializations
    mrf_cost,grad_prior,hessian_prior=qGGMRFfuncs()

    #Function to compute quadratic majorizer for non-linear conjugate gradient inner loop
    #ncg_params = ncg_qGGMRF_funcs()

    vol_size = vol_z*vol_y*vol_x
    proj_size = det_row*det_col*num_views
    
    #Array to save recon
    if 'x_init' in rec_params.keys():
      x_recon = rec_params['x_init'].reshape(vol_size)
      z_recon = rec_params['x_init'].reshape(vol_size)
    else:     
      x_recon = np.ascontiguousarray(np.zeros(vol_size,dtype=np.float32))
      z_recon = np.ascontiguousarray(np.zeros(vol_size,dtype=np.float32))
    
    #Flatten projection data to a vector 
    proj_data = proj_data.reshape(proj_size).astype(np.float32)
    weight_data = weight_data.reshape(proj_size).astype(np.float32)

    #Compute Lipschitz of gradient
    #temp_backproj=np.zeros(vol_size).astype(np.float32)
    temp_backproj=LipschitzForward(vol_size,A,weight_data)
    x_ones = np.ones(vol_size,dtype=np.float32)
    hessian_prior(x_ones,temp_backproj,vol_z,vol_y,vol_x,rec_params['MRF_SIGMA'])
    L = temp_backproj.max()

    #eig_vec,L_f=powerIter(vol_size,A,weight_data,50)
    #del eig_vec
    #L= L + L_f

    #Diagonal majorizer 
    #D= 1.0/temp_backproj
    #D[np.isnan(D)] = 1.0/L
    #D[np.isinf(D)] = 1.0/L
    #print('Min, mean, max of diagonal step size (%f,%f,%f)' %(D.min(),D.mean(),D.max()))
    #D=D*0 + 1.0/L
    
    if(rec_params['verbose']):
      print('Lipschitz constant = %f' %(L))
    del x_ones,temp_backproj
    
    #Initialize variables for Nesterov method
    #ASSUME both x and z are set to zero 
    t_nes = 1
    t=time.time()
    x_prev = np.ascontiguousarray(np.zeros(vol_size,dtype=np.float32))
    gradient = np.ascontiguousarray(np.zeros(vol_size,dtype=np.float32))
    temp_cost = np.zeros(1,dtype=np.float32)
    cost = np.zeros(rec_params['num_iter'])

    #gradient_prev = np.ascontiguousarray(np.zeros(vol_size,dtype=np.float32))
    #cg_dir = np.ascontiguousarray(np.zeros(vol_size,dtype=np.float32))

    #MBIR loop x_k+1 = x_k + func(gradient)
    for iter_num in range(rec_params['num_iter']):

      if(rec_params['verbose']):
        print('Iter number %d of %d in %f sec' %(iter_num,rec_params['num_iter'],time.time()-t))

      #gradient_prev = np.copy(gradient)  #(for SD/NCG)
      error = (A*x_recon)-proj_data
      gradient = A.T*(weight_data*error)
      
      #Cost compute for Debugging
      if rec_params['debug'] == True:
          temp_cost_forward=0.5*(error*weight_data*error).sum()
          mrf_cost(x_recon,temp_cost,vol_z,vol_y,vol_x,rec_params['MRF_P'],rec_params['MRF_SIGMA'])
          cost[iter_num]=temp_cost_forward+temp_cost
          print('Forward Cost %f, Prior Cost %f' %(temp_cost_forward,temp_cost[0]))
          temp_cost = temp_cost*0
          if(iter_num > 0 and (cost[iter_num]-cost[iter_num-1])>0):
              print('Cost went up!')
              t_nes = 1 #reset momentum; adaptive re-start
              
      grad_prior(x_recon,gradient,vol_z,vol_y,vol_x,rec_params['MRF_P'],rec_params['MRF_SIGMA']) #accumulates gradient from prior
      
      x_prev=np.copy(x_recon)
      x_recon,z_recon,t_nes=nesterovOGM2update(x_recon,z_recon,t_nes,gradient,L)

      #if(iter_num ==0): #NCG
        #gradient_prev = np.copy(gradient)  #(for SD/NCG)
        #cg_dir = np.copy(gradient) 
      #NCG
      #x_recon,cg_dir = ncgQMupdate(x_recon,-gradient,-gradient_prev,cg_dir,2,A,weight_data,error,ncg_params,vol_z,vol_x,vol_y,rec_params['MRF_P'],rec_params['MRF_SIGMA'])

      if iter_num>MIN_ITER and stoppingCritVol(x_recon,x_prev,rec_params['stop_thresh'],rec_params['roi_mask']):
          break
      gc.collect() #the call to the C-code grad_prior seems to cause memory to grow; this is a basic fix. TODO: Better memory fix
      
    elapsed_time = time.time()-t
    if(rec_params['verbose']):
      print('Time for %d iterations = %f'%(rec_params['num_iter'],elapsed_time))
    recon= x_recon.reshape(vol_z,vol_y,vol_x)
    return recon,cost

def mbiropTomoTalwar(proj_data,weight_data,A,rec_params):
    """Function for MBIR based on the opTomo function and a Talwar function for log-likelihood term. 
    Inputs: proj_data : A num_rows X num_angles X num_columns array 
            weight_data : A num_rows X num_angles X num_columns array containting the noise variance values 
            A : Spot operator based forward projection matrix 
            rec_params : A dictionary with keys for various parameters of any potential reconstruction algorithm
                       'gpu_index' : Index of GPU to be used 
                       'num_iter' : Number of MBIR iterations 
                       'reg_param' : regulariztion parameter/scale parameter for MRF
                       'reject_frac'   : Threshold for generalized Huber function for likelihood  
    Output : recon : A num_rows X num_columns X num_columns array 
    """
    REJECT_STEP = 5 #Step size for progressive rejection of outliers (0-100)
    NUM_INNER_ITER = 50 #Number of iterations to run with a fixed rejection threshold
    MIN_ITER = NUM_INNER_ITER*REJECT_STEP+10 #Min iter after which to terminate algorithm
    PROGRESSIVE_UPDATE=True
    sigma=1 

    det_row = proj_data.shape[0]
    num_views = proj_data.shape[1]
    det_col = proj_data.shape[2]

    vol_z = rec_params['n_vox_z']
    vol_x = rec_params['n_vox_x']
    vol_y = rec_params['n_vox_y']

    vol_size = vol_z*vol_y*vol_x
    proj_size = det_row*det_col*num_views

    reject_frac = rec_params['reject_frac']

    #Prior model initializations    
    mrf_cost,grad_prior,hessian_prior=qGGMRFfuncs()

    #Flatten projection data to a vector
    proj_data = proj_data.reshape(proj_size).astype(np.float32)
    weight_data = weight_data.reshape(proj_size).astype(np.float32)
    
    #Array to save recon
    #TODO: The logic here has to be fixed. This assumes that if there is in intial input we are in the multi-resolution mode of operation 
    if 'x_init' in rec_params.keys():
      x_recon = rec_params['x_init'].reshape(vol_size)
      z_recon = rec_params['x_init'].reshape(vol_size)
      error = (A*x_recon)-proj_data
      if(rec_params['verbose']):
        print('Target rejection fraction = %f percent' %reject_frac)
      huber_T=np.percentile(np.fabs(error)*np.sqrt(weight_data),100-reject_frac,interpolation='nearest')
      if(x_recon.max() == 0): #TODO: HACK for multi-resolution. At coarsest resolution, use quadratic model
          huber_T=1e5 #Infinite         
      weight_new=np.ascontiguousarray(np.zeros(proj_size,dtype=np.float32))
      weight_new=computeGenHuberWeight(error,sigma,huber_T,0,weight_data,weight_new)
      PROGRESSIVE_UPDATE=False
      if(rec_params['verbose']):
        print('Initializing volume..')
    else:     
      x_recon = np.ascontiguousarray(np.zeros(vol_size,dtype=np.float32))
      z_recon = np.ascontiguousarray(np.zeros(vol_size,dtype=np.float32))
      weight_new = np.ascontiguousarray(np.zeros(proj_size,dtype=np.float32))

    if(rec_params['verbose']):
      print('Starting %d MBIR iterations ..' % (rec_params['num_iter']))

    #Compute Lipschitz of gradient
    x_ones=np.ones(vol_size,dtype=np.float32)
    A_ones= A*x_ones
    temp_backproj = A.T*(weight_data*A_ones) #At*W*A
    L_data = temp_backproj.max()/(sigma**2)
    temp_backproj*=0
    hessian_prior(x_ones,temp_backproj,vol_z,vol_y,vol_x,rec_params['MRF_SIGMA'])
    L_prior = temp_backproj.max()
    L=L_data+L_prior
    if(rec_params['verbose']):
      print('Lipschitz constant = %f' %(L))
    del x_ones,temp_backproj,A_ones
    
    #Initialize variables for Nesterov method
    #ASSUME both x and z are set to zero 
    t_nes = 1
    t=time.time()
    gradient = np.ascontiguousarray(np.zeros(vol_size,dtype=np.float32))
    anomaly_classifier = np.zeros(proj_size,dtype=np.uint8)
    temp_cost = np.zeros(1,dtype=np.float32)
    cost = np.zeros(rec_params['num_iter'])
    momentum = np.zeros(rec_params['num_iter'])
    #MBIR loop x_k+1 = x_k + func(gradient)
    error = (A*x_recon)-proj_data
    for iter_num in range(rec_params['num_iter']):
      if(rec_params['verbose']):
        print('Iter number %d of %d in %f sec' %(iter_num,rec_params['num_iter'],time.time()-t))
      if(np.mod(iter_num,NUM_INNER_ITER)==0 and PROGRESSIVE_UPDATE == True and iter_num <= NUM_INNER_ITER*REJECT_STEP):
        t_nes=1 #Reset momentum 
        curr_rej_frac = np.min([(np.float(iter_num)/NUM_INNER_ITER)*reject_frac/REJECT_STEP,reject_frac])
        if(rec_params['verbose']):
          print('Current rejection fraction = %f' % curr_rej_frac)
        huber_T=np.percentile(np.fabs(error)*np.sqrt(weight_data),100-curr_rej_frac,interpolation='nearest')
        if (iter_num == 0): #First time select all measurements 
            huber_T=5e10

      weight_new=computeTalwarWeight(error,sigma,huber_T,0,weight_data,weight_new) #Compute weight matrix 

      #Cost compute for Debugging
      if rec_params['debug'] == True:
          momentum[iter_num]=t_nes
          temp_cost_forward=computeGenHuberCost(error,weight_new,sigma,huber_T,0)
          mrf_cost(x_recon,temp_cost,vol_z,vol_y,vol_x,rec_params['MRF_P'],rec_params['MRF_SIGMA'])
          cost[iter_num]=temp_cost_forward+temp_cost
          print('Forward Cost %f, Prior Cost %f' %(temp_cost_forward,temp_cost[0]))
          temp_cost = temp_cost*0
          if(iter_num > 0 and (cost[iter_num]-cost[iter_num-1])>0):
              print('Cost went up!')
              
      #Update the volume
      error = (A*x_recon)-proj_data
      gradient = (A.T*(weight_new*error))/sigma**2
      grad_prior(x_recon,gradient,vol_z,vol_y,vol_x,rec_params['MRF_P'],rec_params['MRF_SIGMA']) #accumulates gradient from prior

      x_prev = x_recon
      #Take a step to decrease cost function value w.r.t volume
      x_recon,z_recon,t_nes=nesterovOGM2update(x_recon,z_recon,t_nes,gradient,L)
      if iter_num>MIN_ITER and stoppingCritVol(x_recon,x_prev,rec_params['stop_thresh'],rec_params['roi_mask']):
          print('Number of iterations to convergence %d' %iter_num)
          break 
      gc.collect() #the call to the C-code grad_prior seems to cause memory to grow; this is a basic fix. TODO: Better memory fix

    elapsed_time = time.time()-t
    #if(rec_params['verbose']):
    print('Time for %d iterations = %f'%(rec_params['num_iter'],elapsed_time))

    weight_mask = np.where((np.fabs(error)*np.sqrt(weight_data))>huber_T)
    anomaly_classifier[weight_mask]=1
    anomaly_classifier=anomaly_classifier.reshape(det_row,num_views,det_col)

    recon= x_recon.reshape(vol_z,vol_y,vol_x)
    if rec_params['debug'] == True:
        from matplotlib import pyplot as plt
        plt.plot(momentum);plt.xlabel('Iter');plt.ylabel('Momentum');plt.show()
        
    return recon,cost,anomaly_classifier


def mlCudaDebluropTomo(proj_data,weight_data,A,H,rec_params):
    """Function for GPU based ML estimate based on the Deblur+Project forward. (TODO: Debug)
    Inputs: proj_data : A num_rows X num_angles X num_columns array 
            A         : Forward projection matrix 
            H         : FFT of blur kernel for each view 
            rec_params: Dictionary of parameters associated with the reconstruction algorithm 
    Output : recon : A num_rows X num_cols X num_cols array  
    """
    MIN_ITER = 5
    det_row = proj_data.shape[0]
    num_views = proj_data.shape[1]
    det_col = proj_data.shape[2]

    proj_shape=[det_row,num_views,det_col]

    vol_z = rec_params['n_vox_z']
    vol_x = rec_params['n_vox_x']
    vol_y = rec_params['n_vox_y']

    vol_size = vol_z*vol_y*vol_x
    proj_size = det_row*det_col*num_views
    
    #Array to save recon
    if 'x_init' in rec_params.keys():
      x_recon = rec_params['x_init'].reshape(vol_size)
      z_recon = rec_params['x_init'].reshape(vol_size)
    else:     
      x_recon = np.ascontiguousarray(np.zeros(vol_size,dtype=np.float32))
      z_recon = np.ascontiguousarray(np.zeros(vol_size,dtype=np.float32))
    
    #Flatten projection data to a vector 
    proj_data = proj_data.reshape(proj_size).astype(np.float32)
    weight_data = weight_data.reshape(proj_size).astype(np.float32)

    #Compute Lipschitz of gradient
    #temp_backproj=LipschitzForwardBlurTomo(vol_size,proj_shape,A,H,weight_data)
    #L = temp_backproj.max()
    #del temp_backproj

    eig_val,L=powerIterBlurTomo(vol_size,proj_shape,A,H,weight_data,50)
    del eig_val
    
    if(rec_params['verbose']):
      print('Lipschitz constant = %f' %(L))
    
    #Initialize variables for Nesterov method
    #ASSUME both x and z are set to zero 
    t_nes = 1
    t=time.time()
    gradient = np.ascontiguousarray(np.zeros(vol_size,dtype=np.float32))
    cost = np.zeros(rec_params['num_iter'])
    #ML loop x_k+1 = x_k + func(gradient)
    for iter_num in range(rec_params['num_iter']):
      if(rec_params['verbose']):
        print('Iter number %d of %d in %f sec' %(iter_num,rec_params['num_iter'],time.time()-t))
      error = forwardProject(A,H,x_recon,proj_shape)-proj_data
      gradient = backProject(A,H,weight_data*error,proj_shape)
      #Cost compute for Debugging
      if rec_params['debug'] == True:
          cost[iter_num]=0.5*(error*weight_data*error).sum()
          print('Forward Cost %f' %(cost[iter_num]))
          if(iter_num > 0 and (cost[iter_num]-cost[iter_num-1])>0):
              print('Cost went up!')
      x_prev=np.copy(x_recon)
      #x_recon = gradDescentupdate(x_recon,gradient,1.0/rec_params['step_size'])
      x_recon,z_recon,t_nes=nesterovOGM2update(x_recon,z_recon,t_nes,gradient,L) #1.0/rec_params['step_size'])
      if iter_num>MIN_ITER and stoppingCritVol(x_recon,x_prev,rec_params['stop_thresh'],rec_params['roi_mask']):
              break
          
    elapsed_time = time.time()-t
    if(rec_params['verbose']):
      print('Time for %d iterations = %f'%(rec_params['num_iter'],elapsed_time))
    recon= x_recon.reshape(vol_z,vol_y,vol_x)
    return recon,cost

def mbiropDeblurTomoPoisson(proj_data,weight_data,A,H,rec_params):
    """Function for MBIR based on the opTomo function and a quadratic approximation to the log-likelihood term. This uses less GPU memory but moves large arrays to and from GPU (sub-optimal). Forward model is a tomographic projector followed by a blur kernel  
    Inputs: proj_data : A num_rows X num_angles X num_columns array 
            weight_data : A num_rows X num_angles X num_columns array containting the noise variance values 
            A : Tomographic projector based on ASTRA + spot operator
            H : Blurring kernel of size num_angles X n_x X n_y  
            rec_params : A dictionary with keys for various parameters of any potential reconstruction algorithm
                       'gpu_index' : Index of GPU to be used 
                       'num_iter' : Number of MBIR iterations 
                       'MRF_P' : MRF order parameter
                       'MRF_SIGMA' : regulariztion parameter/scale parameter for MRF
    Output : recon : A num_rows X num_y X num_x array 
    """
    MIN_ITER = 5
    det_row = proj_data.shape[0]
    num_views = proj_data.shape[1]
    det_col = proj_data.shape[2]

    proj_shape=[det_row,num_views,det_col]

    vol_z = rec_params['n_vox_z']
    vol_x = rec_params['n_vox_x']
    vol_y = rec_params['n_vox_y']

    #Prior model initializations
    mrf_cost,grad_prior,hessian_prior=qGGMRFfuncs()

    vol_size = vol_z*vol_y*vol_x
    proj_size = det_row*det_col*num_views
    
    #Array to save recon
    if 'x_init' in rec_params.keys():
      x_recon = rec_params['x_init'].reshape(vol_size)
      z_recon = rec_params['x_init'].reshape(vol_size)
    else:     
      x_recon = np.ascontiguousarray(np.zeros(vol_size,dtype=np.float32))
      z_recon = np.ascontiguousarray(np.zeros(vol_size,dtype=np.float32))
    
    #Flatten projection data to a vector 
    proj_data = proj_data.reshape(proj_size).astype(np.float32)
    weight_data = weight_data.reshape(proj_size).astype(np.float32)

    #Compute upperbound on Lipschitz of gradient
    temp_backproj=np.zeros(vol_size).astype(np.float32) #*LipschitzForwardBlurTomo(vol_size,proj_shape,A,H,weight_data)
    x_ones = np.ones(vol_size,dtype=np.float32)
    hessian_prior(x_ones,temp_backproj,vol_z,vol_y,vol_x,rec_params['MRF_SIGMA'])
    L = temp_backproj.max()

    eig_val,L_f=powerIterBlurTomo(vol_size,proj_shape,A,H,weight_data,50)
    del eig_val
    L+=L_f
    
    if(rec_params['verbose']):
      print('Lipschitz constant = %f' %(L))
    del x_ones,temp_backproj
    
    #Initialize variables for Nesterov method
    #ASSUME both x and z are set to zero 
    t_nes = 1
    t=time.time()
    gradient = np.ascontiguousarray(np.zeros(vol_size,dtype=np.float32))
    temp_cost = np.zeros(1,dtype=np.float32)
    cost = np.zeros(rec_params['num_iter'])
    #MBIR loop x_k+1 = x_k + func(gradient)
    for iter_num in range(rec_params['num_iter']):
      if(rec_params['verbose']):
        print('Iter number %d of %d in %f sec' %(iter_num,rec_params['num_iter'],time.time()-t))
      error = forwardProject(A,H,x_recon,proj_shape)-proj_data
      gradient = backProject(A,H,weight_data*error,proj_shape)
      #Cost compute for Debugging
      if rec_params['debug'] == True:
          temp_cost_forward=0.5*(error*weight_data*error).sum()
          mrf_cost(x_recon,temp_cost,vol_z,vol_y,vol_x,rec_params['MRF_P'],rec_params['MRF_SIGMA'])
          cost[iter_num]=temp_cost_forward+temp_cost
          print('Forward Cost %f, Prior Cost %f' %(temp_cost_forward,temp_cost[0]))
          temp_cost = temp_cost*0
          if(iter_num > 0 and (cost[iter_num]-cost[iter_num-1])>0):
              print('Cost went up!')
              t_nes = 1 #reset momentum; adaptive re-start 
      grad_prior(x_recon,gradient,vol_z,vol_y,vol_x,rec_params['MRF_P'],rec_params['MRF_SIGMA']) #accumulates gradient from prior
      x_prev=x_recon
      x_recon,z_recon,t_nes=nesterovOGM2update(x_recon,z_recon,t_nes,gradient,L)
      if iter_num>MIN_ITER and stoppingCritVol(x_recon,x_prev,rec_params['stop_thresh'],rec_params['roi_mask']):
              break
      gc.collect() #the call to the C-code grad_prior seems to cause memory to grow; this is a basic fix. TODO: Better memory fix
      
    elapsed_time = time.time()-t
    if(rec_params['verbose']):
      print('Time for %d iterations = %f'%(rec_params['num_iter'],elapsed_time))
    recon= x_recon.reshape(vol_z,vol_y,vol_x)
    return recon,cost
   
def mlopTomoPoisson(proj_data,weight_data,A,rec_params):
    """Function for Max. likelihood based on the opTomo function and a quadratic approximation to the log-likelihood term. This uses less GPU memory but moves large arrays to and from GPU (sub-optimal) 
    Inputs: proj_data : A num_rows X num_angles X num_columns array 
            weight_data : A num_rows X num_angles X num_columns array containting the noise variance values 
            A : Spot operator based forward projection matrix 
            rec_params: Dictionary of parameters associated with the reconstruction algorithm 
    Output : recon : A num_rows X num_cols X num_cols array  
    """    
    MIN_ITER = 5
    det_row = proj_data.shape[0]
    num_views = proj_data.shape[1]
    det_col = proj_data.shape[2]

    vol_z = rec_params['n_vox_z']
    vol_x = rec_params['n_vox_x']
    vol_y = rec_params['n_vox_y']
 
    vol_size = vol_z*vol_y*vol_x
    proj_size = det_row*det_col*num_views
    
    #Array to save recon
    if 'x_init' in rec_params.keys():
      x_recon = rec_params['x_init'].reshape(vol_size)
      z_recon = rec_params['x_init'].reshape(vol_size)
    else:     
      x_recon = np.ascontiguousarray(np.zeros(vol_size,dtype=np.float32))
      z_recon = np.ascontiguousarray(np.zeros(vol_size,dtype=np.float32))
    
    #Flatten projection data to a vector 
    proj_data = proj_data.reshape(proj_size).astype(np.float32)
    weight_data = weight_data.reshape(proj_size).astype(np.float32)

    #Compute Lipschitz of gradient

    #temp_backproj=LipschitzForward(vol_size,A,weight_data)
    #L = temp_backproj.max()
    #del temp_backproj

    eig_val,L=powerIter(vol_size,A,weight_data,50)
    del eig_val 
    
    if(rec_params['verbose']):
      print('Lipschitz constant = %f' %(L))
    
    #Initialize variables for Nesterov method
    #ASSUME both x and z are set to zero 
    t_nes = 1
    t=time.time()
    x_prev = np.ascontiguousarray(np.zeros(vol_size,dtype=np.float32))
    gradient = np.ascontiguousarray(np.zeros(vol_size,dtype=np.float32))
    temp_cost = np.zeros(1,dtype=np.float32)
    cost = np.zeros(rec_params['num_iter'])

    #ML loop x_k+1 = x_k + func(gradient)
    for iter_num in range(rec_params['num_iter']):

      if(rec_params['verbose']):
        print('Iter number %d of %d in %f sec' %(iter_num,rec_params['num_iter'],time.time()-t))

      error = (A*x_recon)-proj_data
      gradient = A.T*(weight_data*error)
      
      #Cost compute for Debugging
      if rec_params['debug'] == True:
          temp_cost_forward=0.5*(error*weight_data*error).sum()
          cost[iter_num]=temp_cost_forward+temp_cost
          print('Forward Cost %f' %(temp_cost_forward))
          temp_cost = temp_cost*0
          if(iter_num > 0 and (cost[iter_num]-cost[iter_num-1])>0):
              print('Cost went up!')
              t_nes = 1 #reset momentum; adaptive re-start
              
      x_prev=np.copy(x_recon)
      x_recon,z_recon,t_nes=nesterovOGM2update(x_recon,z_recon,t_nes,gradient,L)

      if iter_num>MIN_ITER and stoppingCritVol(x_recon,x_prev,rec_params['stop_thresh'],rec_params['roi_mask']):
          break
      
    elapsed_time = time.time()-t
    if(rec_params['verbose']):
      print('Time for %d ML iterations = %f'%(rec_params['num_iter'],elapsed_time))
    recon= x_recon.reshape(vol_z,vol_y,vol_x)
    return recon,cost
