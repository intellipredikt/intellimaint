# 10 steps input, 10 steps output

import numpy as np
from IntelliMaint.file_handler import get_data
from minisom import MiniSom
import matplotlib.pyplot as plt
import pickle 
from sklearn import preprocessing
import GPy

class BearingPrognostics:
	def __init__(self):
		self.scaler = preprocessing.MinMaxScaler()
		third_set_3_rms, third_set_3_kurtosis, third_set_3_crest = get_data()
		input_vectors_3 = np.concatenate((third_set_3_rms, third_set_3_kurtosis, third_set_3_crest), axis=1)
		self.input_vectors_3 = self.scaler.fit_transform(input_vectors_3)

	def get_train_input_vector(self):
		input_vectors_3_ = self.input_vectors_3[:4000]
		return input_vectors_3_

	def get_test_input_vector(self):
		test_input_vector_3_ = (self.input_vectors_3)[5650:]
		return test_input_vector_3_

	def quantization_error(self, qnt_arr, input_vector_arr):
		qe = [] # quantization error
		for i in range(len(qnt_arr)):
			qe.append(np.linalg.norm(qnt_arr[i] - input_vector_arr[i])) 
		return np.array(qe)


	def make_gpr_data(self, qe): # 10-dim input, 10-dim output
		result = np.zeros((1, 10)) # initialize numpy array
		y = []
		for i in range(len(qe)-10):
			temp_list = []
			for j in range(i, i+10):
				temp_list.append(qe[j])
			array = np.array(temp_list)
			result = np.concatenate((result, array.reshape(1, 10)), axis=0)

		y = result[2:] # one-step shifted down

		return result[1:-1], y

	def train_som(self):
		train_X = self.get_train_input_vector()

		self.som = MiniSom(50, 50, 3, sigma=0.1, learning_rate=0.5) # a 6x6 SOM
		self.som.random_weights_init(train_X)
		starting_weights = self.som.get_weights().copy()  # saving the starting weights
		self.som.train_random(train_X, 500) # 500 iterations

		print('quantization...')
		qnt = self.som.quantization(train_X)  # quantize each vector of the train_X
		print('building new image...')
		clustered = np.zeros(train_X.shape)
		for i, q in enumerate(qnt):  # place the quantized values into a new image
			clustered[i] = q
		print('done.')

		# save som model and weights
		# saving the som in the file som.p
		with open('som.p', 'wb') as outfile:
		    pickle.dump(self.som, outfile)

	def train_gpr(self):
		test_3 = self.get_test_input_vector() # has degradation area

		qnt_test_3 = self.som.quantization(test_3)

		qe_test_3 = self.quantization_error(qnt_test_3, test_3)

		# preparing dataset for GPR
		gpr_test_X, gpr_test_Y = self.make_gpr_data(qe_test_3)

		self.train_X, self.train_Y = gpr_test_X[:100], gpr_test_Y[:100]
		self.test_X, self.test_Y = gpr_test_X[100:163], gpr_test_Y[100:163]

		self.kernel = GPy.kern.Matern32(input_dim=10, variance=1, lengthscale=1.)
		self.kernel += GPy.kern.RBF(input_dim=10, variance=1, lengthscale=1.)
		self.kernel += GPy.kern.RatQuad(input_dim=10, variance=1, lengthscale=1.0)
		self.m = GPy.models.GPRegression(self.train_X, self.train_Y, self.kernel)
		self.m.optimize()

	def predict_rul(self):

		Yp, Vp = self.m.predict(self.test_X)
		plotter_pred = []
		plotter_true = []
		plotter_threshold = [0.3 for i in range(len(self.test_X))]
		plotter_critical = [0.5 for i in range(len(self.test_X))]
		for i in range(len(Yp)):
			plotter_pred.append(Yp.squeeze()[i][0])
			plotter_true.append(self.test_Y.squeeze()[i][0])

		plt.plot(plotter_pred, label='predicted')
		plt.plot(plotter_true, label='true')
		plt.plot(plotter_threshold, linestyle='dashed', label='threshold')
		plt.plot(plotter_critical, linestyle='dotted', label='critical')
		plt.legend()
		plt.xlabel('timesteps')
		plt.ylabel('MQE')
		plt.show()

