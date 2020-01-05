import GPy
import numpy as np

class DegradationModel:
	def __init__(self, HI):
		print (HI)
		if (HI.shape[0] == 1):
			HI = HI.reshape(1, 1)
			self.kernel = GPy.kern.Poly(input_dim=HI.shape[0])
			timesteps = np.array([i for i in range(len(HI))]).reshape(len(HI), HI.shape[0])

		else:
			self.kernel = GPy.kern.Poly(input_dim=HI.shape[1])
			timesteps = np.array([i for i in range(len(HI))]).reshape(len(HI), HI.shape[1])
		self.gpmodel = GPy.models.GPRegression(timesteps, HI, self.kernel)
		self.optimize()

	def optimize(self):
		self.gpmodel.optimize()

	def update(self, X, Y):
		self.gpmodel = GPy.models.GPRegression(X, Y, self.kernel)
		self.optimize()

	def predict(self, X):
		Yp, Vp = self.gpmodel.predict(X)
		return Yp.squeeze(), Vp.squeeze()