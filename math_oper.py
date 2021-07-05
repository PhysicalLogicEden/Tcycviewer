# -*- coding: utf-8 -*-
"""
Created on Sun May 30 16:35:16 2021
Performs math calculations/operators
"""
#%%Import libs
import numpy as np


#%%Smooth
"""
Smoothing Data by Rolling Average with NumPy
inputs:
    data        = 1-D input array
    smooth_val  = window width for smoothing
outputs:
    data_convolved    = smoothed data
"""  
def smooth(data, smooth_val):
    kernel_size = smooth_val # equals to smooth by smooth_val
    kernel = np.ones(kernel_size) / kernel_size
    data_convolved = np.convolve(data, kernel, mode='same')
    data_convolved[0] = np.average(data[:smooth_val])
    for i in range(1, smooth_val):
        data_convolved[i] = np.average(data[:i + smooth_val])
        data_convolved[-i] = np.average(data[-i - smooth_val:])
    return data_convolved


#%% Savitzky–Golay filter

# data_smoothed = savgol_filter(dataTcyc[:,0],51,3)
