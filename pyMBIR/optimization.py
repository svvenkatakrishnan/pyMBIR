# Copyright (C) 2019, S.V.Venkatakrishnan <venkatakrisv@ornl.gov>
# All rights reserved. BSD 3-clause License.
# This file is part of the pyMBIR package. Details of the copyright
# and user license can be found in the 'LICENSE' file distributed
# with the package.

import numpy as np 

#Functions to optimize regularized inverse problems (for tomography)


def gradDescentupdate(x,grad,L):
    """
    Gradient descent update with step size 1/L
    """
    xNew = x - grad/L
    return xNew

def nesterovOGM1update(x,z,t,grad,L):
    """
    Nestrov OGM1 update 
    See.D. Kim and J. A. Fessler, "An optimized first-order method for image restoration," 2015 IEEE International Conference on Image Processing (ICIP), Quebec City, QC, 2015, pp. 3675-3679. doi: 10.1109/ICIP.2015.7351490
    """
    zNew = x - grad/L
    tNew = 0.5*(1+np.sqrt(1+4*(t**2)))
    xNew = zNew + ((t-1)/tNew)*(zNew-z)
    return xNew,zNew,tNew

def nesterovOGM2update(x,z,t,grad,L):
    """
    Nestrov OGM2 update
    See.D. Kim and J. A. Fessler, "An optimized first-order method for image restoration," 2015 IEEE International Conference on Image Processing (ICIP), Quebec City, QC, 2015, pp. 3675-3679. doi: 10.1109/ICIP.2015.7351490
    """
    zNew = x  - grad/L
    tNew = 0.5*(1+np.sqrt(1+4*(t**2)))
    xNew = (1+((t -1)/tNew))*zNew - ((t -1)/tNew)*z - (t/tNew)*grad/L #zNew + ((t -1)/tNew)*(zNew-z) + (t/tNew)*(-grad/L)
    return xNew,zNew,tNew

def ncgQMupdate(x,d,d_p,h_p,num_iter,A,W,error_init,ncg_params,num_slice,num_row,num_col,mrf_p,mrf_sigma):
    """
    Non-linear conjugate gradient with quadratic majorization for inner loop 
    Inputs: x : current estimate 
            d : Negative of Current gradient of cost function 
            d_p: Negative of Previous gradient 
            h_p : Previous search direction 
            num_iter : Number of iterations for majorize minimize
            y : Data array  
            A : forward model operator (ASTRA spot operator)
            W : Weight matrix 
           error_init : Ax-y Initial error (from gradient compute)
            ncg_params: Function call to quadratic majorizer coefficients 
            num_slice,num_row,num_col: Volume parameters 
            mrf_p,mrf_sigma: QGGMRF Prior model parameters 
    """
    gamma = (np.dot(d-d_p,d)/np.dot(d_p,d)) #Polak-Riberie Step size for new direction; if set to zero - steepest descent
    gamma = np.max([0.0,gamma]) #Restarting (Page 42:https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf) 
    h = d + gamma*h_p
    theta2_temp = A*h
    theta2_temp2 = W*theta2_temp
    theta2_init = np.dot(theta2_temp,theta2_temp2) #TODO: Approximate because AT not equal to A but comp. cheap
    theta1_init = np.array(np.dot(error_init,theta2_temp2),dtype=np.float32)
    alpha = 0
    del theta2_temp,theta2_temp2 
    for k in range(num_iter):
        theta1_init += alpha*theta2_init
        theta1 = np.copy(theta1_init) 
        theta2 = np.array(theta2_init,dtype=np.float32)
        ncg_params(x,h,theta1,theta2,num_slice,num_row,num_col,mrf_p,mrf_sigma)
        alpha = np.max([0.0,-theta1/theta2])
        x += alpha*h
 
    return x,h


