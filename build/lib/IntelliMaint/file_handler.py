# data set : 2nd_test 
# column 0 : Bearing 1 (outer race failure)
# column 1 : Bearing 2 (normal)
# column 2 : Bearing 3 (normal)
# column 3 : Bearing 4 (normal)

# data set : 3rd_test
# column 0 : Bearing 1 (normal)
# column 1 : Bearing 2 (normal)
# column 2 : Bearing 3 (outer race failure)
# column 3 : Bearing 4 (normal)

import numpy as np
from scipy.stats import kurtosis
import os
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)
def get_raw(directory):
	i = 0 
	data1 = None
	data2 = None
	for file in sorted(os.listdir(directory)):
		if i == 130:
			data1 = np.loadtxt(directory+'/'+file)[:, 0]
		if i == 5967:
			data2 = np.loadtxt(directory+'/'+file)[:, 0]
		i += 1

	return data1, data2

def get_kurtosis(X): 
	"""
	Args: 
		X -- 1-D numpy array
	"""

	return kurtosis(X)

def get_rms(X):
	"""
	Args:
		X -- 1-D numpy array
	"""
	rms = np.sqrt(np.mean(X**2))
	return rms

def get_crest(X):
	peak = max(X)
	rms = get_rms(X)
	return np.divide(peak, rms)

def get_data(directory, feature, bearing_num):
	list_data = []
	for file in sorted(os.listdir(directory)):
		if (feature == 'rms'):
			feature_value = get_rms((np.loadtxt(directory+'/'+file))[:, bearing_num-1])
		elif (feature == 'kurtosis'):
			feature_value = get_kurtosis((np.loadtxt(directory+'/'+file))[:, bearing_num-1])
		else:
			feature_value = get_crest((np.loadtxt(directory+'/'+file))[:, bearing_num-1])
		list_data.append(feature_value)
	return np.array(list_data).reshape(len(list_data), 1)

def save_all_data():
	# from 1st_test
	# 3rd bearing channel 6 (column 6)
	first_test_bearing_3_rms = get_data('1st_test', 'rms', 6)
	first_test_bearing_3_kurtosis = get_data('1st_test', 'kurtosis', 6)
	first_test_bearing_3_crest = get_data('1st_test', 'crest', 6)
	np.save('first_test_bearing_3_rms.npy', first_test_bearing_3_rms)
	np.save('first_test_bearing_3_kurtosis.npy', first_test_bearing_3_kurtosis)
	np.save('first_test_bearing_3_crest.npy', first_test_bearing_3_crest)
	
	# # from 2nd_test
	# second_test_bearing_1_rms = get_data('2nd_test', 'rms', 1)
	# second_test_bearing_1_kurtosis = get_data('2nd_test', 'kurtosis', 1)
	# second_test_bearing_1_crest = get_data('2nd_test', 'crest', 1)
	# np.save('2nd_test_bearing_1_rms.npy', second_test_bearing_1_rms[500:, :])
	# np.save('2nd_test_bearing_1_kurtosis.npy', second_test_bearing_1_kurtosis[500:, :])
	# np.save('2nd_test_bearing_1_crest.npy', second_test_bearing_1_crest[500:, :])
	# second_test_bearing_2_rms = get_data('2nd_test', 'rms', 2)
	# second_test_bearing_2_kurtosis = get_data('2nd_test', 'kurtosis', 2)
	# second_test_bearing_2_crest = get_data('2nd_test', 'crest', 2)
	# np.save('2nd_test_bearing_2_rms.npy', second_test_bearing_2_rms[500:, :])
	# np.save('2nd_test_bearing_2_kurtosis.npy', second_test_bearing_2_kurtosis[500:, :])
	# np.save('2nd_test_bearing_2_crest.npy', second_test_bearing_2_crest[500:, :])
	# second_test_bearing_3_rms = get_data('2nd_test', 'rms', 3)
	# second_test_bearing_3_kurtosis = get_data('2nd_test', 'kurtosis', 3)
	# second_test_bearing_3_crest = get_data('2nd_test', 'crest', 3)
	# np.save('2nd_test_bearing_3_rms.npy', second_test_bearing_3_rms[500:, :])
	# np.save('2nd_test_bearing_3_kurtosis.npy', second_test_bearing_3_kurtosis[500:, :])
	# np.save('2nd_test_bearing_3_crest.npy', second_test_bearing_3_crest[500:, :])
	# second_test_bearing_4_rms = get_data('2nd_test', 'rms', 4)
	# second_test_bearing_4_kurtosis = get_data('2nd_test', 'kurtosis', 4)
	# second_test_bearing_4_crest = get_data('2nd_test', 'crest', 4)
	# np.save('2nd_test_bearing_4_rms.npy', second_test_bearing_4_rms[500:, :])
	# np.save('2nd_test_bearing_4_kurtosis.npy', second_test_bearing_4_kurtosis[500:, :])
	# np.save('2nd_test_bearing_4_crest.npy', second_test_bearing_4_crest[500:, :])

	# # from 3rd_test
	# third_test_bearing_1_rms = get_data('3rd_test', 'rms', 1)
	# third_test_bearing_1_kurtosis = get_data('3rd_test', 'kurtosis', 1)
	# third_test_bearing_1_crest = get_data('3rd_test', 'crest', 1)
	# np.save('3rd_test_bearing_1_rms.npy', third_test_bearing_1_rms[500:, :])
	# np.save('3rd_test_bearing_1_kurtosis.npy', third_test_bearing_1_kurtosis[500:, :])
	# np.save('3rd_test_bearing_1_crest.npy', third_test_bearing_1_crest[500:, :])
	# third_test_bearing_2_rms = get_data('3rd_test', 'rms', 2)
	# third_test_bearing_2_kurtosis = get_data('3rd_test', 'kurtosis', 2)
	# third_test_bearing_2_crest = get_data('3rd_test', 'crest', 2)
	# np.save('3rd_test_bearing_2_rms.npy', third_test_bearing_2_rms[500:, :])
	# np.save('3rd_test_bearing_2_kurtosis.npy', third_test_bearing_2_kurtosis[500:, :])
	# np.save('3rd_test_bearing_2_crest.npy', third_test_bearing_2_crest[500:, :])
	# third_test_bearing_3_rms = get_data('3rd_test', 'rms', 3)
	# third_test_bearing_3_kurtosis = get_data('3rd_test', 'kurtosis', 3)
	# third_test_bearing_3_crest = get_data('3rd_test', 'crest', 3)
	# np.save('3rd_test_bearing_3_rms.npy', third_test_bearing_3_rms[500:, :])
	# np.save('3rd_test_bearing_3_kurtosis.npy', third_test_bearing_3_kurtosis[500:, :])
	# np.save('3rd_test_bearing_3_crest.npy', third_test_bearing_3_crest[500:, :])
	# third_test_bearing_4_rms = get_data('3rd_test', 'rms', 4)
	# third_test_bearing_4_kurtosis = get_data('3rd_test', 'kurtosis', 4)
	# third_test_bearing_4_crest = get_data('3rd_test', 'crest', 4)
	# np.save('3rd_test_bearing_4_rms.npy', third_test_bearing_4_rms[500:, :])
	# np.save('3rd_test_bearing_4_kurtosis.npy', third_test_bearing_4_kurtosis[500:, :])
	# np.save('3rd_test_bearing_4_crest.npy', third_test_bearing_4_crest[500:, :])

def get_data():
	third_set_1_rms = np.load(dirname+'/numpy_data/3rd_test_bearing_1_rms.npy', allow_pickle=True)
	third_set_1_kurtosis= np.load(dirname+'/numpy_data/3rd_test_bearing_1_kurtosis.npy', allow_pickle=True)
	third_set_1_crest = np.load(dirname+'/numpy_data/3rd_test_bearing_1_crest.npy', allow_pickle=True)

	third_set_2_rms = np.load(dirname+'/numpy_data/3rd_test_bearing_2_rms.npy', allow_pickle=True)
	third_set_2_kurtosis = np.load(dirname+'/numpy_data/3rd_test_bearing_2_kurtosis.npy', allow_pickle=True)
	third_set_2_crest = np.load(dirname+'/numpy_data/3rd_test_bearing_2_crest.npy', allow_pickle=True)

	third_set_3_rms = np.load(dirname+'/numpy_data/3rd_test_bearing_3_rms.npy', allow_pickle=True)
	third_set_3_kurtosis = np.load(dirname+'/numpy_data/3rd_test_bearing_3_kurtosis.npy', allow_pickle=True)
	third_set_3_crest = np.load(dirname+'/numpy_data/3rd_test_bearing_3_crest.npy', allow_pickle=True)

	third_set_4_rms = np.load(dirname+'/numpy_data/3rd_test_bearing_4_rms.npy', allow_pickle=True)
	third_set_4_kurtosis = np.load(dirname+'/numpy_data/3rd_test_bearing_4_kurtosis.npy', allow_pickle=True)
	third_set_4_crest = np.load(dirname+'/numpy_data/3rd_test_bearing_4_crest.npy', allow_pickle=True)

	return third_set_3_rms, third_set_3_kurtosis, third_set_3_crest

def get_3rd_bearing_data():
	rms = np.load('numpy_data/first_test_bearing_3_rms.npy', allow_pickle=True)
	kurtosis = np.load('numpy_data/first_test_bearing_3_kurtosis.npy', allow_pickle=True)
	crest = np.load('numpy_data/first_test_bearing_3_crest.npy', allow_pickle=True)
	return rms, kurtosis, crest

# X_train, X_test = get_train_test_data()
# plt.plot(X_train, label='train')
# plt.plot(X_test, label='test')
# plt.legend()
# plt.show()
# save_all_data()

# output1, output2 = get_raw('3rd_test')


# plt.figure(figsize=(7, 7))
# plt.figure(1)
# plt.subplot(221)
# plt.title('normal')
# plt.plot(output1)
# plt.subplot(222)
# plt.title('abnormal')
# plt.plot(output2)
# plt.tight_layout()
# plt.show()