import GPy
import numpy as np
from skrvm import RVR
import matplotlib.pyplot as plt

class GPRDegradationModel:
	def __init__(self, HI, failure_threshold, order=1):
		x = np.array([i for i in range(len(HI))]).reshape(-1, 1)
		y = HI
		print (x.shape, y.shape)
		self.kernel = GPy.kern.Poly(input_dim=HI.shape[1], variance=1., scale=1., bias=1., order=order)
		self.gpmodel = GPy.models.GPRegression(x, y, self.kernel)
		self.optimize()
		self.current_iteration = 0
		self.failure_threshold = failure_threshold

	def optimize(self):
		self.gpmodel.optimize()

	def update(self, X, Y):
		self.gpmodel = GPy.models.GPRegression(X, Y, self.kernel)
		self.optimize()
		self.current_iteration += 1

	def predict(self, X):
		Yp, Vp = self.gpmodel.predict(X)
		Yp, Vp = Yp.squeeze(), Vp.squeeze()
		failure_idx, = np.where(Yp >= self.failure_threshold)
		rul = None
		if (len(failure_idx) > 0):
			first_failure_idx = failure_idx[0]
			rul = first_failure_idx - self.current_iteration

		return Yp, Vp, rul

class RVRDegradationModel:
	def __init__(self, HI):
		if (HI.shape[0] == 1):
			HI = HI.reshape(1, 1)
			timesteps = np.array([i for i in range(len(HI))]).reshape(len(HI), HI.shape[0])
		else:
			timesteps = np.array([i for i in range(len(HI))]).reshape(len(HI), HI.shape[1])
		self.rvrmodel = RVR(kernel='linear')
		self.optimize(timesteps, HI)

	def optimize(self, X, Y):
		self.rvrmodel.fit(X, Y)

	def update(self, X, Y):
		self.optimize(X, Y)

	def predict(self, X):
		# self.rvrmodel.fit(X, X)
		Yp = self.rvrmodel.predict(X)
		print (Yp)
		return Yp

