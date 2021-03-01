
/*
 Copyright (C) 2019, S.V.Venkatakrishnan <venkatakrisv@ornl.gov>
 All rights reserved. GPL v3 license.
 This file is part of the pyMBIR package. Details of the copyright
 and user license can be found in the 'LICENSE' file distributed
 with the package.
 */
/*Code to compute the gradient of an image based on a 3-D MRF*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <omp.h>
#include "mrf3d_grad_linear.h"
#include <time.h>


int32_t wrap(int32_t index,int32_t max_val)
{
  if(index<0)
    return index+max_val;
  else
    if(index>=max_val)
      return index%max_val;
    else
      return index;
}

int64_t IDX(int64_t slice,int64_t row,int64_t col,int64_t NROW,int64_t NCOL) 
{
  int64_t index;
  index = slice*NROW*NCOL + row*NCOL + col;
  return index;
}

/*The regularizing MRF potential*/
float pot_func(float delta, float MRF_P, float MRF_SIGMA)
{
  return ((powf(fabsf(delta)/MRF_SIGMA,MRF_Q))/(MRF_C + powf(fabsf(delta)/MRF_SIGMA,MRF_Q - MRF_P)));
}

/*The first derivative of the potential function*/
float deriv_potFunc(float delta,float MRF_P,float MRF_SIGMA,float MRF_SIGMA_X_Q)
{
  float temp1,temp2,temp3;
  float abs_delta = fabsf(delta);
  temp1=powf(abs_delta/MRF_SIGMA,MRF_Q - MRF_P);
  temp2=powf(abs_delta,MRF_Q - 1);
  temp3 = MRF_C + temp1;
  if(delta < 0.0)
    return ((-1*temp2/(temp3*MRF_SIGMA_X_Q))*(MRF_Q - ((MRF_Q-MRF_P)*temp1)/(temp3)));
  else if (delta > 0.0)
    return ((temp2/(temp3*MRF_SIGMA_X_Q))*(MRF_Q - ((MRF_Q-MRF_P)*temp1)/(temp3)));
  else return 0;
}

/*Second Derivative of the potential function at zero */ 
float second_deriv_potFunc_zero(float MRF_SIGMA_X_Q)
{
  return MRF_Q/(MRF_SIGMA_X_Q*MRF_C);
}


void mrf_grad(float* in_img,float* out_img, int NUM_SLICE,int NUM_ROW, int NUM_COL,float MRF_P,float MRF_SIGMA)
{
  int64_t i,j,k,p,q,r;  
  float diff;
  int64_t num_slice,num_row,num_col;
  float SIGMA_X_Q;
  float *temp_img;

  num_slice=(int64_t)NUM_SLICE;
  num_row=(int64_t)NUM_ROW;
  num_col=(int64_t)NUM_COL;  
  SIGMA_X_Q = powf(MRF_SIGMA, MRF_Q);
		   
#pragma omp parallel for collapse(3) private(i,j,k,p,q,r,diff)
  for(k=0;k < num_col;k++)
  for (j=0;j < num_row;j++)
  for (i=0; i < num_slice;i++)
	{
         for (p=-1;p<2;p++)
	   for (q=-1;q<2;q++)
	     for (r=-1;r<2;r++)	       
      	       if(i+p >=0 && i+p <= NUM_SLICE-1 && j+q >=0 && j+q <= NUM_ROW-1 && k+r >= 0 && k+r <= NUM_COL-1)
		 {
		   //if(in_img[IDX(i,j,k,num_row,num_col)] != in_img[IDX(i+p,j+q,k+r,num_row,num_col)]){
		  diff = in_img[IDX(i,j,k,num_row,num_col)]-in_img[IDX(i+p,j+q,k+r,num_row,num_col)];
		  out_img[IDX(i,j,k,num_row,num_col)]+=(FILTER[p+1][q+1][r+1]*deriv_potFunc(diff,MRF_P,MRF_SIGMA,SIGMA_X_Q));    
		  //}
		 }
	}
  
}

void mrf_diag_Hessian_zero(float* in_img,float* out_img, int NUM_SLICE,int NUM_ROW, int NUM_COL,float MRF_SIGMA)
{
  int64_t i,j,k,p,q,r,num_row,num_col,num_slice;  
  float diff;
  float SIGMA_X_Q = powf(MRF_SIGMA, MRF_Q);
  float val_zero=second_deriv_potFunc_zero(SIGMA_X_Q);
  num_slice=(int64_t)NUM_SLICE;
  num_row=(int64_t)NUM_ROW;
  num_col = (int64_t)NUM_COL;
#pragma omp parallel for collapse(3) private(i,j,k,p,q,r) 
  for (i=0; i < num_slice;i++)
    for (j=0;j < num_row;j++)
      for(k=0;k < num_col;k++)
         for (p=-1;p<2;p++)
	   for (q=-1;q<2;q++)
	     for (r=-1;r<2;r++)
	       if(i+p >=0 && i+p <= NUM_SLICE-1 && j+q >=0 && j+q <= NUM_ROW-1 && k+r >= 0 && k+r <= NUM_COL-1)
	       {
		 out_img[IDX(i,j,k,num_row,num_col)]+=(FILTER[p+1][q+1][r+1]*val_zero);
	       }
}

void mrf_cost(float* in_img,float* cost_val, int NUM_SLICE,int NUM_ROW, int NUM_COL,float MRF_P,float MRF_SIGMA)
{
  int64_t i,j,k,p,q,r,num_row,num_col,num_slice;  
  float delta,temp;
  temp=0;
  num_slice=(int64_t)NUM_SLICE;
  num_row=(int64_t)NUM_ROW;
  num_col = (int64_t)NUM_COL;
#pragma omp parallel for collapse(3) private(delta, i, j, k) reduction(+:temp)
  for (i = 0; i < num_row; i++)
    {
      for (j = 0; j < num_col; j++)
	{
	  for (k = 0; k < num_slice; k++)
	    {
	      if(k + 1 < num_slice)
		{
		  delta = in_img[IDX(k,i,j,num_row,num_col)] - in_img[IDX(k+1,i,j,num_row,num_col)];
		  temp += FILTER[2][1][1]*pot_func(delta,MRF_P,MRF_SIGMA);
		}
	      if(j + 1 < num_col)
		{
		  if(k - 1 >= 0)
		    {
		      delta = in_img[IDX(k,i,j,num_row,num_col)] - in_img[IDX(k-1,i,j+1,num_row,num_col)];
		      temp += FILTER[0][1][2]*pot_func(delta,MRF_P,MRF_SIGMA);
		    }

		  delta = in_img[IDX(k,i,j,num_row,num_col)] - in_img[IDX(k,i,j+1,num_row,num_col)];
		  temp += FILTER[1][1][2] * pot_func(delta, MRF_P,MRF_SIGMA);

		  if(k + 1 < num_slice)
		    {
		      delta = in_img[IDX(k,i,j,num_row,num_col)] - in_img[IDX(k+1,i,j+1,num_row,num_col)];
		      temp += FILTER[2][1][2] * pot_func(delta, MRF_P,MRF_SIGMA);
		    }
		}

	      if(i+1 < num_row)
		{
		  if(j-1 >= 0)
		    {
		      delta = in_img[IDX(k,i,j,num_row,num_col)] - in_img[IDX(k,i+1,j-1,num_row,num_col)];
		      temp += FILTER[1][2][0] * pot_func(delta, MRF_P,MRF_SIGMA);
		    }

		  delta = in_img[IDX(k,i,j,num_row,num_col)] - in_img[IDX(k,i+1,j,num_row,num_col)];
		  temp += FILTER[1][2][1] * pot_func(delta, MRF_P,MRF_SIGMA);

		  if(j + 1 < num_col)
		    {
		      delta = in_img[IDX(k,i,j,num_row,num_col)] - in_img[IDX(k,i + 1, j + 1,num_row,num_col)];
		      temp += FILTER[1][2][2] * pot_func(delta, MRF_P,MRF_SIGMA);
		    }

		  if(j - 1 >= 0)
		    {
		      if(k - 1 >= 0)
			{
			  delta = in_img[IDX(k,i,j,num_row,num_col)] - in_img[IDX(k-1,i + 1, j - 1, num_row,num_col)];
			  temp += FILTER[0][2][0] * pot_func(delta, MRF_P,MRF_SIGMA);
			}

		      if(k + 1 < num_slice)
			{
			  delta = in_img[IDX(k,i,j,num_row,num_col)] - in_img[IDX(k+1,i + 1, j - 1,num_row,num_col)];
			  temp += FILTER[2][2][0] * pot_func(delta, MRF_P,MRF_SIGMA);
			}

		    }

		  if(k - 1 >= 0)
		    {
		      delta = in_img[IDX(k,i,j,num_row,num_col)] - in_img[IDX(k-1,i + 1,j,num_row,num_col)];
		      temp += FILTER[0][2][1] * pot_func(delta, MRF_P,MRF_SIGMA);
		    }

		  if(j + 1 < num_col)
		    {
		      if(k - 1 >= 0)
			{
			  delta = in_img[IDX(k,i,j,num_row,num_col)] - in_img[IDX(k-1,i+1,j+1,num_row,num_col)];
			  temp += FILTER[0][2][2] * pot_func(delta, MRF_P,MRF_SIGMA);
			}

		      if(k + 1 < num_slice)
			{
			  delta = in_img[IDX(k,i,j,num_row,num_col)] - in_img[IDX(k+1,i+1,j+1,num_row,num_col)];
			  temp += FILTER[2][2][2] * pot_func(delta, MRF_P,MRF_SIGMA);
			}
		    }

		  if(k + 1 < num_slice)
		    {
		      delta = in_img[IDX(k,i,j,num_row,num_col)] - in_img[IDX(k+1,i + 1, j, num_row,num_col)];
		      temp += FILTER[2][2][1] * pot_func(delta, MRF_P,MRF_SIGMA);
		    }
		}
	    }
	}
    }
  *cost_val += (temp);
  
}

/*
Parameters for inner loops of non-linear conjugate gradient based on a quadratic surrogate function approach (instead of line search)
*/
void ncg_inner_params(float* in_img,float* search_dir, float *theta1, float *theta2, int NUM_SLICE,int NUM_ROW, int NUM_COL,float MRF_P,float MRF_SIGMA)
{
  int64_t i,j,k,p,q,r,num_row,num_col,num_slice,index1,index2;  
  float delta_x,delta_d,temp,temp2,temp_var;
  float SIGMA_X_Q;
  temp=0;
  temp2=0;
  num_slice=(int64_t)NUM_SLICE;
  num_row=(int64_t)NUM_ROW;
  num_col = (int64_t)NUM_COL;
  SIGMA_X_Q = powf(MRF_SIGMA, MRF_Q);
#pragma omp parallel for collapse(3) private(delta_x,delta_d, i, j, k) reduction(+:temp,temp2)
  for (i = 0; i < num_row; i++)
    {
      for (j = 0; j < num_col; j++)
	{
	  for (k = 0; k < num_slice; k++)
	    {
	      index1=IDX(k,i,j,num_row,num_col);
	      if(k + 1 < num_slice)
		{
		  index2=IDX(k+1,i,j,num_row,num_col);
		  delta_x = in_img[index1] - in_img[index2];
		  delta_d = search_dir[index1] - search_dir[index2];
		  if (delta_x != 0.0)
		    temp_var = FILTER[2][1][1]*deriv_potFunc(delta_x,MRF_P,MRF_SIGMA,SIGMA_X_Q)*delta_d/(2*delta_x);
		  else
		    temp_var = FILTER[2][1][1]*second_deriv_potFunc_zero(SIGMA_X_Q)*delta_d/2;
		    
		  temp += temp_var*delta_d;
		  temp2 += temp_var*delta_x;
		  
		}
	      if(j + 1 < num_col)
		{
		  if(k - 1 >= 0)
		    {
		      index2=IDX(k-1,i,j+1,num_row,num_col);
		      delta_x = in_img[index1] - in_img[index2];
		      delta_d = search_dir[index1] - search_dir[index2];
		      if (delta_x != 0.0)
			temp_var = FILTER[0][1][2]*deriv_potFunc(delta_x,MRF_P,MRF_SIGMA,SIGMA_X_Q)*delta_d/(2*delta_x);
		      else
			temp_var = FILTER[0][1][2]*second_deriv_potFunc_zero(SIGMA_X_Q)*delta_d/2;

		      temp += temp_var*delta_d;
		      temp2 += temp_var*delta_x;
		    }
		  index2=IDX(k,i,j+1,num_row,num_col);
		  delta_x = in_img[index1] - in_img[index2];
		  delta_d = search_dir[index1] - search_dir[index2];
		  if (delta_x != 0.0)
		    temp_var = FILTER[1][1][2] * deriv_potFunc(delta_x, MRF_P,MRF_SIGMA,SIGMA_X_Q)*delta_d/(2*delta_x);
		  else
		    temp_var = FILTER[1][1][2]*second_deriv_potFunc_zero(SIGMA_X_Q)*delta_d/2;

		  temp += temp_var*delta_d;
		  temp2 += temp_var*delta_x;

		  if(k + 1 < num_slice)
		    {
		      index2 = IDX(k+1,i,j+1,num_row,num_col);
		      delta_x = in_img[index1] - in_img[index2];
		      delta_d = search_dir[index1] - search_dir[index2];
		      if (delta_x != 0.0)
			temp_var= FILTER[2][1][2] * deriv_potFunc(delta_x, MRF_P,MRF_SIGMA,SIGMA_X_Q)*delta_d/(2*delta_x);
		      else
			temp_var = FILTER[2][1][2]*second_deriv_potFunc_zero(SIGMA_X_Q)*delta_d/2;

		      temp += temp_var*delta_d;
		      temp2 += temp_var*delta_x;
		   
		    }
		}

	      if(i+1 < num_row)
		{
		  if(j-1 >= 0)
		    {
		      index2=IDX(k,i+1,j-1,num_row,num_col);
		      delta_x = in_img[index1] - in_img[index2];
		      delta_d = search_dir[index1] - search_dir[index2];
		      if (delta_x != 0.0)
			temp_var = FILTER[1][2][0] * deriv_potFunc(delta_x, MRF_P,MRF_SIGMA,SIGMA_X_Q)*delta_d/(2*delta_x);
		      else
			temp_var = FILTER[1][2][0]*second_deriv_potFunc_zero(SIGMA_X_Q)*delta_d/2;

		      temp += temp_var*delta_d;
		      temp2 += temp_var*delta_x;
		      
		    }
		  index2=IDX(k,i+1,j,num_row,num_col);
		  delta_x = in_img[index1] - in_img[index2];
		  delta_d = search_dir[index1] - search_dir[index2];

		  if (delta_x != 0.0)
		    temp_var = FILTER[1][2][1] *deriv_potFunc(delta_x, MRF_P,MRF_SIGMA,SIGMA_X_Q)*delta_d/(2*delta_x);
		  else
		    temp_var = FILTER[1][2][1]*second_deriv_potFunc_zero(SIGMA_X_Q)*delta_d/2;

		  temp += temp_var*delta_d;
		  temp2 += temp_var*delta_x;
		  
		  if(j + 1 < num_col)
		    {
		      index2=IDX(k,i + 1, j + 1,num_row,num_col);
		      delta_x = in_img[index1] - in_img[index2];
		      delta_d = search_dir[index1] - search_dir[index2];
	    	      if (delta_x != 0.0)
			temp_var = FILTER[1][2][2] * deriv_potFunc(delta_x, MRF_P,MRF_SIGMA,SIGMA_X_Q)*delta_d/(2*delta_x);
		      else
			temp_var = FILTER[1][2][2]*second_deriv_potFunc_zero(SIGMA_X_Q)*delta_d/2;

		      temp += temp_var*delta_d;
		      temp2 += temp_var*delta_x;		      
		    }

		  if(j - 1 >= 0)
		    {
		      if(k - 1 >= 0)
			{
			  index2=IDX(k-1,i + 1, j - 1, num_row,num_col);
			  delta_x = in_img[index1] - in_img[index2];
		    	  delta_d = search_dir[index1] - search_dir[index2];
			  if (delta_x != 0.0)
			    temp_var = FILTER[0][2][0] * deriv_potFunc(delta_x, MRF_P,MRF_SIGMA,SIGMA_X_Q)*delta_d/(2*delta_x);
			  else
			    temp_var = FILTER[0][2][0]*second_deriv_potFunc_zero(SIGMA_X_Q)*delta_d/2;

			  temp += temp_var*delta_d;
			  temp2 += temp_var*delta_x;		      			  
			}

		      if(k + 1 < num_slice)
			{
			  index2=IDX(k+1,i + 1, j - 1,num_row,num_col);
			  delta_x = in_img[index1] - in_img[index2];
			  delta_d = search_dir[index1] - search_dir[index2];
			  if (delta_x != 0.0)
			    temp_var = FILTER[2][2][0] * deriv_potFunc(delta_x, MRF_P,MRF_SIGMA,SIGMA_X_Q)*delta_d/(2*delta_x);
			  else
			    temp_var = FILTER[2][2][0]*second_deriv_potFunc_zero(SIGMA_X_Q)*delta_d/2;

			  temp += temp_var*delta_d;
			  temp2 += temp_var*delta_x;
			}
		    }

		  if(k - 1 >= 0)
		    {
		      index2=IDX(k-1,i + 1,j,num_row,num_col);
		      delta_x = in_img[index1] - in_img[index2];
		      delta_d = search_dir[index1] - search_dir[index2];
	       	      if (delta_x != 0.0)
			temp_var = FILTER[0][2][1] * deriv_potFunc(delta_x, MRF_P,MRF_SIGMA,SIGMA_X_Q)*delta_d/(2*delta_x);
		      else
			temp_var = FILTER[0][2][1]*second_deriv_potFunc_zero(SIGMA_X_Q)*delta_d/2;

		      temp += temp_var*delta_d;
		      temp2 += temp_var*delta_x;
		    }

		  if(j + 1 < num_col)
		    {
		      if(k - 1 >= 0)
			{
			  index2=IDX(k-1,i+1,j+1,num_row,num_col);
			  delta_x = in_img[index1] - in_img[index2];
			  delta_d = search_dir[index1] - search_dir[index2];
			  if (delta_x != 0.0)
			    temp_var = FILTER[0][2][2] * deriv_potFunc(delta_x, MRF_P,MRF_SIGMA,SIGMA_X_Q)*delta_d/(2*delta_x);
			  else
			    temp_var = FILTER[0][2][2]*second_deriv_potFunc_zero(SIGMA_X_Q)*delta_d/2;

			  temp += temp_var*delta_d;
			  temp2 += temp_var*delta_x;
			}

		      if(k + 1 < num_slice)
			{
			  index2=IDX(k+1,i+1,j+1,num_row,num_col);
			  delta_x = in_img[index1] - in_img[index2];
			  delta_d = search_dir[index1] - search_dir[index2];
			  if (delta_x != 0.0)
			    temp_var = FILTER[2][2][2] * deriv_potFunc(delta_x, MRF_P,MRF_SIGMA,SIGMA_X_Q)*delta_d/(2*delta_x);
			  else
			    temp_var = FILTER[2][2][2]*second_deriv_potFunc_zero(SIGMA_X_Q)*delta_d/2;

			  temp += temp_var*delta_d;
			  temp2 += temp_var*delta_x;

			  
			}
		    }

		  if(k + 1 < num_slice)
		    {
		      index2=IDX(k+1,i + 1, j, num_row,num_col);
		      delta_x = in_img[index1] - in_img[index2];
		      delta_d = search_dir[index1] - search_dir[index2];
		      if (delta_x != 0.0)
			temp_var = FILTER[2][2][1] * deriv_potFunc(delta_x, MRF_P,MRF_SIGMA,SIGMA_X_Q)*delta_d/(2*delta_x);
		      else
			temp_var = FILTER[2][2][1]*second_deriv_potFunc_zero(SIGMA_X_Q)*delta_d/2;

		      temp += temp_var*delta_d;
		      temp2 += temp_var*delta_x;
		    }
		}
	    }
	}
    }
  *theta1 += 2*temp2;
  *theta2 += 2*temp;
  
}

void main()
{

  int32_t num_slice=2;
  int32_t num_row=4;
  int32_t num_col = 4;
  float *inp_img,*out_img;
  float cost;
  int32_t i;
  float MRF_P = 1.2;
  float MRF_SIGMA = 0.001;
  float time_start,time_elapsed;

  inp_img = (float*)malloc(num_slice*num_row*num_col*sizeof(float));
  out_img =  (float*)malloc(num_slice*num_row*num_col*sizeof(float));
  cost=0;

  for(i=0;i<num_slice*num_row*num_col;i++)
    inp_img[i]=((float)rand()/(float)RAND_MAX);

  printf("Starting compute ..\n");
  time_start=clock();
  mrf_grad(inp_img,out_img, num_slice,num_row, num_col,MRF_P,MRF_SIGMA);
  time_elapsed=clock()-time_start;
  printf("Time taken for gradient=%f seconds\n",(float)(time_elapsed)/CLOCKS_PER_SEC/100);

  time_start=clock();
  mrf_diag_Hessian_zero(inp_img,out_img,num_slice,num_row,num_col,MRF_SIGMA);
  time_elapsed=clock()-time_start;
  printf("Time taken for Hessian=%f seconds\n",(float)(time_elapsed)/CLOCKS_PER_SEC/100);

  time_start=clock();
  mrf_cost(inp_img,&cost,num_slice,num_row, num_col,MRF_P,MRF_SIGMA);
  time_elapsed=clock()-time_start;
  printf("Time taken for cost=%f seconds\n",(float)(time_elapsed)/CLOCKS_PER_SEC/100);

  
}
