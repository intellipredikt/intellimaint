#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 20:01:57 2020

@author: anup
"""
import pandas as pd
import os as os
import numpy as np
import sys
import statistics as st
import scipy as sp
import scipy.stats as sps
from scipy.stats import kurtosis
from scipy.stats import skew

#%% Time Domian Class
class TimeDomain:
    
    # def __init__(self, data):
    #     self.data = data
        
    def get_rms(self,data):
        if data.ndim == 3:
            rms = np.sqrt(np.mean(data**2, axis = 1))
        else:
            rms = np.sqrt(np.mean(data**2, axis = 0))
        return rms
        
    def get_mean(self,data):
        if data.ndim == 3:
            men = np.mean(data, axis = 1)
        else:
            men = np.mean(data, axis = 0)
        return men
    
    def get_variance(self,data):
        if data.ndim == 3:
            v = np.var(data, axis =1)
        else:
            v = np.var(data, axis =0)
        return v
        
        
    def get_crestfactor(self,data):
        if data.ndim == 3:
            peaks = np.max(data, axis = 1)
            rms = self.get_rms(data)
            c = np.divide(peaks, rms)
        else:
            peaks = np.max(data, axis = 0)
            rms = self.get_rms(data)
            c = np.divide(peaks, rms)
        return c
        
        
        
    def get_kurtosis(self,data):
        if data.ndim == 3:
            kurt = kurtosis(data, axis=1)
        else:
            kurt = kurtosis(data, axis=0)
        return kurt
        
        
    def get_skewness(self,data):
        if data.ndim == 3:
            sk = skew(data, axis = 1)
        else:
            sk = skew(data, axis = 0)
        return sk
        
        
#%% Frequency Domain (stationary)
        
class FrequencyDomain:
    
    def __init__(self,data):
        self.data = data
        
    def get_cepstrumcoeffs(self,data):
        spectrum = np.fft.fft(data,axis=1)
        ceps_coeffs = np.fft.ifft(np.log(np.abs(spectrum))).real
        return ceps_coeffs

        
   
    # def get_spectralanalysis(self,data):
        
        
#%% Time - Frequency (Non Stationary )
        

class Nonstationary:
    
    
    def __init__(self,data):
        self.data = data
      
    #Emphirical Mode Decomposition
    # def get_emd(self, data):
        
    #Wavelet Packet Decomposition
    # def get_wpd(self, data):
     
        
        
        
    