/*
 Copyright (C) 2019, S.V.Venkatakrishnan <venkatakrisv@ornl.gov>
 All rights reserved. GPL v3 license.
 This file is part of the pyMBIR package. Details of the copyright
 and user license can be found in the 'LICENSE' file distributed
 with the package.
 */

float pot_func(float delta, float MRF_P, float MRF_SIGMA);
float deriv_potFunc(float delta,float MRF_P,float MRF_SIGMA,float MRF_SIGMA_X_Q);
float second_deriv_potFunc_zero(float MRF_SIGMA_X_Q);
void mrf_grad(float *in_img,float *out_img,int NSLICE,int NROW,int NCOL,float MRF_P,float MRF_SIGMA);
void mrf_diag_Hessian_zero(float* in_img,float* out_img, int NUM_SLICE,int NUM_ROW, int NUM_COL,float MRF_SIGMA);
void mrf_cost(float* in_img,float* cost_val, int NUM_SLICE,int NUM_ROW, int NUM_COL,float MRF_P,float MRF_SIGMA);
void ncg_inner_params(float* in_img,float* search_dir, float *theta1, float *theta2, int NUM_SLICE,int NUM_ROW, int NUM_COL,float MRF_P,float MRF_SIGMA);
int64_t IDX(int64_t slice,int64_t row,int64_t col,int64_t NROW,int64_t NCOL);

float MRF_C=0.001;
float MRF_Q = 2;
const float FILTER[3][3][3]={{{0.0302,0.0370,0.0302},{0.0370,0.0523,0.0370},{0.0302,0.0370,0.0302}},
		       {{0.0370,0.0523,0.0370},{0.0523,0.0,0.0523},   {0.0370,0.0523,0.0370}},
		       {{0.0302,0.0370,0.0302},{0.0370,0.0523,0.0370},{0.0302,0.0370,0.0302}}};
