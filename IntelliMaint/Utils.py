import numpy as np
import os, glob
import re
import pandas as pd
import json

dirname = os.path.dirname(__file__)

class Utils:
	def __init__(self):
		self.turboengine_idx = 1
		self.bearing_idx = 4

	def get_features(self, component_type):

		def sorted_aphanumeric(data):
		    convert = lambda text: int(text) if text.isdigit() else text.lower()
		    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
		    return sorted(data, key=alphanum_key)

		def state_of_health(data):
			initial_capacity = data[0]
			for i in range(len(data)):
				data[i] = (data[i]/initial_capacity)*100
			return data

		def get_battery_data():
			data_dir='examples/data/battery_data/B0006_discharge.json'
			data_dir = dirname + '/' + data_dir
			with open(data_dir) as f:    
			    discharge_data = json.load(f)
			data = []
			time_stamps = []
			for cycle in discharge_data.keys():
				data.append(discharge_data[cycle]["capacity"][0]) # remove the last dimension
				time_stamps.append(discharge_data[cycle]["date_time"])
			soh = np.array(state_of_health(data)).reshape(len(data), 1)
			times = np.array(time_stamps).reshape(len(time_stamps), 1)
			return soh

		if (component_type == 'turboengine'):
			data_dir='examples/data/phm08_data/csv'
			data_dir = dirname + '/' + data_dir
			dirlist = sorted_aphanumeric(os.listdir(data_dir))
			all_data = []
			for file in dirlist:
				if (not file.endswith('.py')) and (file != '__pycache__'):
					data = pd.read_csv(data_dir+"/"+file, engine='python')
					data = np.array(data)
					all_data.append(data[:, 2:])
			all_data_np = np.array(all_data)
			return all_data_np[self.turboengine_idx]
		elif (component_type == 'bearing'):
			data_dir='examples/data/bearing_data/'
			data_dir = dirname + '/' + data_dir
			rms = np.load(data_dir+'/3rd_test_bearing_'+str(self.bearing_idx)+'_rms.npy', allow_pickle=True)
			kurtosis = np.load(data_dir+'/3rd_test_bearing_'+str(self.bearing_idx)+'_kurtosis.npy', allow_pickle=True)
			crest = np.load(data_dir+'/3rd_test_bearing_'+str(self.bearing_idx)+'_crest.npy', allow_pickle=True)
			all_data_np = np.concatenate((rms, kurtosis, crest), axis=1)
			return all_data_np

		elif (component_type == 'battery'):
			return get_battery_data()
			
	def check1(self, data):
		assert isinstance(data,np.ndarray),'*** Please provide numpy array as input ***'
		assert len(data.shape)>0,'*** Ensure the array is not empty ***'
		assert np.all(np.isfinite(data)),'*** Remove NaN and Inf values from the numpy array'

	# Count the number of positive peaks
	def count_pos(self, data):
	    if(len(data.shape) == 1):
	        count = (data>0).sum()
	    else:
	        count = [(column>0).sum() for column in data.T]
	    return np.asarray(count)

	def count_neg(self, data):
	    if(len(data.shape) == 1):
	        count = (data<0).sum()
	    else:
	        count = [(column<0).sum() for column in data.T]
	    return np.asarray(count)


	def zscore(self, data, mu, sigma):
		z_s = (data - mu)/sigma
		return z_s    

	def zs_param(self, train, percentileScore=None):
		if(percentileScore == None):
			mu, sigma = np.mean(train), np.std(train)
		else:
			mu, sigma = np.percentile(train,percentileScore), np.std(train)
		return mu, sigma


