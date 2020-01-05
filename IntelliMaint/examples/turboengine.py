import numpy as np
from IntelliMaint.rul_models import DegradationModel
from IntelliMaint.health_indicators import FeatureFusion
import matplotlib.pyplot as plt
from IntelliMaint import Utils

class TurboEngineRULModel(FeatureFusion, DegradationModel):
	def __init__(self, train_HI_data, obs):
		FeatureFusion.__init__(self, train_HI_data) # first fifty to train som
		HI = self.health_indicator(obs)
		DegradationModel.__init__(self, HI)
		self.HI = HI
		self.obs = obs

	def observe(self, X): # new raw observation
		self.obs = np.concatenate((self.obs, X), axis=0)
		self.HI = self.health_indicator(self.obs) # new observations
		return self.HI

	def predict(self, X):
		Yp, Vp = DegradationModel.predict(self, X)
		return Yp, Vp

	def update(self):
		l = len(self.obs)
		X = np.array([i for i in range(l)]).reshape(l, 1)
		DegradationModel.update(self, X, self.HI)

utils = Utils() # i-th data-set out of 218 data-sets

# step 1: Get Features
all_observations = utils.get_features('turboengine')

# step 2: Identify the health indicator
# use the initial data (normal) for feature fusion
old_obs = all_observations[:120] # assuming first 50 datapoints are normal
new_obs = all_observations[120:]

train_HI_data = old_obs[:100]
obs = old_obs[100:]

model = TurboEngineRULModel(train_HI_data, obs) # use initial 100 obs to prepare model

last_observation_index = 19 # from obs

next_steps = 5

for j in range(12):
	X = np.array([i for i in range(last_observation_index+next_steps+1)]).reshape(last_observation_index+next_steps+1, 1)
	Yp, _ = model.predict(X)
	HI = model.observe(new_obs[j*next_steps:(j+1)*next_steps])
	model.update()
	plt.plot(Yp)
	plt.scatter(X, model.HI, label='observations', color='k', marker='o', alpha=0.5)
	plt.show()
	last_observation_index += 5