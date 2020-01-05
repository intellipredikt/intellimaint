from minisom import MiniSom
import scipy.signal
from sklearn import preprocessing
import numpy as np

class FeatureFusion:
	def __init__(self, X):
		self.minmaxscaler = preprocessing.StandardScaler()
		X = self.minmaxscaler.fit_transform(X)
		self.som = MiniSom(50, 50, X.shape[1], sigma=0.1, learning_rate=0.5)
		self.som.random_weights_init(X)
		starting_weights = self.som.get_weights().copy()  # saving the starting weights
		self.som.train_random(X, 500) # 500 iterations

	def quantization_error(self, qnt_arr, input_vector_arr):
		qe = [] # quantization error
		for i in range(len(qnt_arr)):
			qe.append(np.linalg.norm(qnt_arr[i] - input_vector_arr[i])) 
		return np.array(qe)

	def health_indicator(self, X):
		X = self.minmaxscaler.transform(X)
		qnt_som_X = self.som.quantization(X)
		qe = self.quantization_error(qnt_som_X, X)
		self.HI_raw = qe
		qe_filtered = scipy.signal.medfilt(qe, 11) # median filter
		hi = scipy.signal.savgol_filter(qe_filtered, 11, 1)
		return hi.reshape(len(hi), 1)

class FeatureIndicator:
	def __init__(self):
		pass

	def health_indicator(self, X):
		return X