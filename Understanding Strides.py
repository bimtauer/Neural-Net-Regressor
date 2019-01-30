# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 09:36:22 2019

@author: bimta
"""
import numpy as np
from numpy.lib.stride_tricks import as_strided


A = np.arange(12.).reshape(3, -1)

#one stride = 8 for floats, 4 for ints

# Reproducing the array:
as_strided(A, shape = (3,4), strides = (32,8))


# Every other col:
as_strided(A, shape = (3,2), strides = (32,16))

# Every other row:
as_strided(A, shape = (2,4), strides = (64,8))

# Moving window size 3 -> on 3x4 array that yields 2 3x3 matrixes, shifted 8 bytes between another, 32 bytes in each row, and 8 inside a column
as_strided(A, shape = (2,3,3), strides = (8,32,8))


A = np.arange(12.)

# One dimensional moving window k = 3 -> fits 10 times in the original array
as_strided(A, shape = (10, 3), strides = (8,8))
