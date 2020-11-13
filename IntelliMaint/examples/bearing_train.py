# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:32:17 2020

@author: Admin
"""
import os
from IntelliMaint.eda import ExploratoryAnalysis
from IntelliMaint.feature_engineering import TimeDomain
from IntelliMaint.data_analysis import SOM
from IntelliMaint.health_assessment import HealthIndicator
from IntelliMaint.anomaly_detection import AnomalyDetection
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
import pickle as pkl

analysis_info = {}

#----------Exploratory Data Analysis-----------------------#

files = sorted(glob.glob('data/bearing_data/3rd_test/txt/200*'))

# for i in np.arange(len(files)):
# 	data = pd.read_csv(files[i],sep='\t',header=None)
# 	e = ExploratoryAnalysis(data)
# 	e.perform_eda()

# # ----------Feature Extraction------------------------------#
# a = []
# td = TimeDomain()
# ber=1
# for i in range(len(files)):
#     df = pd.read_csv(files[i],sep='\t',header = None, names=(['0','1','2','3'])) 
#     c = np.reshape([td.get_rms(df[str(ber)]),td.get_kurtosis(df[str(ber)]),\
#     	td.get_mean(df[str(ber)]),td.get_crestfactor(df[str(ber)]),\
#     	td.get_skewness(df[str(ber)]),td.get_variance(df[str(ber)])],\
#     	(-1,6))
#     a.append(c)
#     fea=np.array(a).reshape(-1,6)
       
# out=pd.DataFrame(fea,columns=['rms','kurtosis','mean','crestfactor','skewness','variance'])
# out['File']=files
# out.set_index('File', inplace = True)
# out.to_csv('data/temp/Bearing_1_3rd.csv')
# del out

# #----------Data Analysis/State Detection------------------#
data = pd.read_csv('data/temp/Bearing_1_3rd.csv')
data = np.array(data)[:, 1:]
f1, f2, f3, f4, f5, f6 = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5]

# use rms, kurtosis, crest for fusion
data = np.concatenate((f1.reshape(len(f1), 1), f2.reshape(len(f2), 1), f4.reshape(len(f4), 1)), axis=1)
train_som = data[:1000]
test_som = data
da = SOM()
som, scaler = da.train(train_som)
som_error = da.predict(som, test_som, scaler) # on full data

# rms, kurtosis, crest, error
data = np.concatenate((data, som_error.reshape(len(som_error), 1)), axis=1)

analysis_info['extracted_features'] = ['rms', 'kurtosis', 'crest', 'error']
analysis_info['normal_data_pts'] = 1000
analysis_info['prediction_horizon'] = 100
# #-----------Health Assessment-------------#
ad = HealthIndicator()
score = ad.computeHIScore(data)
# col = np.argmax(score)
col = 3

hi = data[:, col]

print ('HI Scores :', score, ' | Selected HI :', col)

analysis_info['health_indicator'] = analysis_info['extracted_features'][col]

hi_train = hi[:1000]
hi_test = hi
ad = AnomalyDetection()
hi_train = pd.DataFrame(hi_train)
ad.train_cosmo(hi_train)

hi_test = pd.DataFrame(hi_test)
strangeness, _ = ad.test_cosmo(hi_test)
strangeness = strangeness.squeeze()

analysis_info['min_continuous_deg_pts'] = 10
analysis_info['incipient_fault_threshold'] = 1.0
analysis_info['failure_threshold'] = 3.0

print ('Saving Analysis Information in data/temp/analysis_info.pkl')
f = open("data/temp/analysis_info.pkl", "wb")
pkl.dump(analysis_info, f)
f.close()








