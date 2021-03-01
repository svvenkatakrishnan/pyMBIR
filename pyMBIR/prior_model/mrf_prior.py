# Copyright (C) 2019, S.V.Venkatakrishnan <venkatakrisv@ornl.gov>
# All rights reserved. GPL v3 license.
# This file is part of the pyMBIR package. Details of the copyright
# and user license can be found in the 'LICENSE' file distributed
# with the package.
 
import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer

def qGGMRFfuncs():
    '''
    Prior model initializations and interface to C functions 
    '''
#    lib_path='mrf3d_grad_linear.so'
#    lib = ctypes.cdll.LoadLibrary(lib_path)
    from .. import lib
    grad_prior = lib.mrf_grad
    grad_prior.restype = None
    grad_prior.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_float]

    hessian_prior = lib.mrf_diag_Hessian_zero #Hessian at zero 
    hessian_prior.restype = None
    hessian_prior.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_float]

    mrf_cost = lib.mrf_cost
    mrf_cost.restype = None
    mrf_cost.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_float]
    return mrf_cost,grad_prior,hessian_prior

def ncg_qGGMRF_funcs():
    '''
    Prior model initializations and interface to C functions for non-linear conjugate gradient
    '''
    from .. import lib
    ncg_params = lib.ncg_inner_params
    ncg_params.restype = None
    ncg_params.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_float]
    return ncg_params
