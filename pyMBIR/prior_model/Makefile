# Copyright (C) 2017 by S. V. Venkatakrishnan (venkatakrisv@ornl.gov)
# All rights reserved. BSD 3-clause License.
# This file is part of the tomoORNL package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

CC = gcc
CFLAGS = -fPIC -fopenmp -shared -O3 -lm #Compiler flags. Can change the code optimization levels.
TARGET = mrf3d_grad_linear.so 
SRCS = mrf3d_grad_linear.c
HDRS = mrf3d_grad_linear.h
OBJECTS = $(SRCS:.c=.o)

all : $(TARGET)

$(TARGET): $(OBJECTS)   
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS)
clean:
	rm -f *.o *.so 
