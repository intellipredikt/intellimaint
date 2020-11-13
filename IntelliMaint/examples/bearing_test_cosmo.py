# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 

@author: IPTLP0018
"""

import os
from IntelliMaint import Utils
from IntelliMaint.feature_engineering import TimeDomain
from IntelliMaint.data_analysis import SOM
from IntelliMaint.anomaly_detection import AnomalyDetection
from IntelliMaint.health_assessment import HealthIndicator
from IntelliMaint.rul_models import GPRDegradationModel
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
import scipy.signal

analysis_info = pd.read_pickle('data/temp/analysis_info.pkl')
# print (analysis_info)

'''
{
'prediction_horizon': 5324, 
'extracted_features': ['rms', 'kurtosis', 'crest', 'error'], 
'normal_data_pts': 1000, 
'health_indicator': 'error', 
'min_continuous_deg_pts': 10, 
'incipient_fault_threshold': 1.0
'failure_threshold': 5.0
}
'''

# prediction_horizon = 5324
prediction_horizon = analysis_info['prediction_horizon']
normal_data_pts = analysis_info['normal_data_pts']
min_continuous_deg_pts = analysis_info['min_continuous_deg_pts']
incipient_fault_threshold = analysis_info['incipient_fault_threshold']
failure_threshold = analysis_info['failure_threshold']



files = sorted(glob.glob('data/bearing_data/3rd_test/txt/200*'))

data_list = []
td = TimeDomain()
test_bearing_idx = 0

# wait for initial 1000 observations
i = 0
while (i <= normal_data_pts):
	df = pd.read_csv(files[i],sep='\t',header = None, names=(['0','1','2','3'])) 
	temp = np.reshape([td.get_rms(df[str(test_bearing_idx)]),\
		td.get_kurtosis(df[str(test_bearing_idx)]),\
		td.get_crestfactor(df[str(test_bearing_idx)])],\
		(-1,3))
	data_list.append(temp)
	i += 1

initial_features = np.array(data_list).reshape(-1,3)
print ("Done. 1000 samples.")
#-----------Data Analysis-------------------------------------
# use rms, kurtosis, crest for fusion
da = SOM()
som, scaler = da.train(initial_features)
train_error = da.predict(som, initial_features, scaler)

ad = AnomalyDetection()
train_error = pd.DataFrame(train_error)
ad.train_cosmo(train_error)

# #-----------Anomaly Detection & Health Assessment--------------
initial_deg_pts = []
iter_= 0
tracker = 0
flag = False
continuously_deg_pts = 0
deg_start_idx = 0
for i in range(normal_data_pts+1, len(files)):
	df = pd.read_csv(files[i],sep='\t',header = None, names=(['0','1','2','3'])) 
	fea = np.reshape([td.get_rms(df[str(test_bearing_idx)]),\
		td.get_kurtosis(df[str(test_bearing_idx)]),\
		td.get_crestfactor(df[str(test_bearing_idx)])],(-1,3))
	
	error = da.predict(som, fea, scaler).reshape(1, 1)
	error = pd.DataFrame(error)
	strangeness, _ = ad.test_cosmo(error)
	strangeness = strangeness.squeeze()
	
	# check for continuous degradation
	if (strangeness >= incipient_fault_threshold) and (not flag):
		deg_start_idx = i
		iter_ += 1
		flag = True
	elif (strangeness >= incipient_fault_threshold):
		iter_ += 1

	if (flag):
		tracker += 1

		if (iter_ / tracker == 1.0):
			continuously_deg_pts += 1
			initial_deg_pts.append(strangeness)
		else:
			initial_deg_pts = []
			continuously_deg_pts = 0
			iter_ = 0
			tracker = 0
			flag = False
	
	if (continuously_deg_pts >= min_continuous_deg_pts):
		break

# #-----------RUL Estimation-------------------------------------
hi_raw = np.array(initial_deg_pts).reshape(len(initial_deg_pts), 1)
rul_model = GPRDegradationModel(hi_raw, failure_threshold, order=1)
for i in range(deg_start_idx+min_continuous_deg_pts, len(files)):
	df = pd.read_csv(files[i],sep='\t',header = None, names=(['0','1','2','3'])) 
	fea = np.reshape([td.get_rms(df[str(test_bearing_idx)]),\
		td.get_kurtosis(df[str(test_bearing_idx)]),\
		td.get_crestfactor(df[str(test_bearing_idx)])],(-1,3))
	
	error = da.predict(som, fea, scaler).reshape(1, 1)
	error = pd.DataFrame(error)
	strangeness, _ = ad.test_cosmo(error)
	hi_raw = np.concatenate((hi_raw, strangeness.reshape(-1, 1)), axis=0)

	# predict next prediction_horizon intervals
	Yp, Vp, rul = rul_model.predict(np.array([k for k in range(min_continuous_deg_pts+1, min_continuous_deg_pts+1+prediction_horizon)]).reshape(prediction_horizon, 1))
	print (rul)
	# update the model
	x = np.array([k for k in range(len(hi_raw))]).reshape(len(hi_raw), 1)
	y = hi_raw
	rul_model.update(x, y)

	plt.plot(Yp)
	plt.plot(hi_raw[min_continuous_deg_pts:])
	plt.fill_between(np.array([k for k in range(len(Yp))]).reshape(len(Yp), 1).squeeze(), Yp, Yp + Vp, color='k', alpha=.5)
	plt.fill_between(np.array([k for k in range(len(Yp))]).reshape(len(Yp), 1).squeeze(), Yp, Yp - Vp, color='k', alpha=.5)
	plt.plot([failure_threshold for i in range(len(Yp))], linestyle='dashed', color='#ff0000')
	plt.show()


	


