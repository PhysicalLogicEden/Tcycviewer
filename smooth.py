# -*- coding: utf-8 -*-
"""
Created on Sun May 30 16:35:16 2021
Performs Smooth function
inputs:
    data        = 1-D input array
    smooth_val  = window width for smoothing
outputs:
    data_sm     = smoothed data
"""


def smooth(data, smooth_val):    
    kernel_size = 100 # equals to smooth by 100
    kernel = np.ones(kernel_size) / kernel_size
    data_convolved = np.convolve(dataTcyc[:,0], kernel, mode='same')    
