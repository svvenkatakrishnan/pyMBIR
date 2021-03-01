#Phantoms for generating simulated data sets

import numpy as np 

def cube_phantom(num_slice,im_size,cube_z,cube_y,cube_x,density):
    #Create a simple cube phantom
    #num_slice is the total number of slices in the volume
    #im_size number of rows/columns in the volume
    #cube_z,cube_y,cube_x : size of the actual phantom 
    obj = np.zeros((num_slice,im_size,im_size)).astype(np.float32)
    y,x=np.ogrid[-im_size/2:im_size/2,-im_size/2:im_size/2]
    height_idx = slice(num_slice//2-cube_z//2,num_slice//2+cube_z//2)
    col_idx = slice(im_size//2-cube_y//2,im_size//2+cube_y//2)
    row_idx = slice(im_size//2-cube_x//2,im_size//2+cube_x//2)
    obj[height_idx,row_idx,col_idx]=density
    return obj
