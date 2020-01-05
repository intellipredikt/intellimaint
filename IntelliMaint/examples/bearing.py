import numpy as np
from IntelliMaint.rul_models import DegradationModel
from IntelliMaint.health_indicators import FeatureFusion
import matplotlib.pyplot as plt
from IntelliMaint import Utils

class BearingRULModel(FeatureFusion, DegradationModel):
	def __init__(self, train_HI_data, obs):
		FeatureFusion.__init__(self, train_HI_data) # first fifty to train som
		HI = self.health_indicator(obs)
		self.HI = HI
		self.obs = obs
		self.HI_to_train_model = None
		self.t_incipient = 0 # incipient fault time
		self.t_current = 0

	def observe(self, X): # new raw observation
		self.obs = np.concatenate((self.obs, X), axis=0)
		self.HI = self.health_indicator(self.obs) # new observations
		return self.HI

	def predict(self, next_steps):
		l = self.t_current + next_steps
		X = np.array([i for i in range(l)]).reshape(l, 1)
		Yp, Vp = DegradationModel.predict(self, X)
		self.t_current += next_steps
		return Yp, Vp

	def initialize_model(self):
		DegradationModel.__init__(self, self.HI[self.t_incipient])

	def update(self):
		l = self.t_current
		X = np.array([i for i in range(self.t_current)]).reshape(l, 1)
		DegradationModel.update(self, X, self.HI[self.t_incipient:self.t_incipient+self.t_current])
		

utils = Utils() # i-th data-set out of 218 data-sets

# step 1: Get Features
all_observations = utils.get_features('bearing')

# step 2: Identify the health indicator
# use the initial data (normal) for feature fusion
old_obs = all_observations[:600] # assuming these datapoints are normal
new_obs = all_observations[600:]

train_HI_data = old_obs[:500]
obs = old_obs[500:]

model = BearingRULModel(train_HI_data, obs) # use initial 100 obs to prepare model
next_steps = 20

out = model.observe(new_obs[:5155].reshape(len(new_obs[:5155]), 3))

# x = np.array([(i*10)/60 for i in range(500, 500+len(model.HI))])
# plt.plot(x, model.HI_raw/10.0, label='HI', color='g')
# plt.plot(x, model.HI/10.0, label='Filtered HI', color='k')
# plt.plot(x, [0.1907940 for i in range(len(model.HI))], linestyle='dashed', color='#FFA500')
# plt.plot(x, [1.0 for i in range(len(model.HI))], linestyle='dashed', color='#ff0000')

# plt.title('Health Indicator (HI)')
# plt.xlabel('timesteps (hours)')
# plt.ylabel('Health')
# plt.legend()
# ax = plt.gca()
# ax.set_facecolor((0.827, 0.827, 0.827))
# plt.show()

for j in range(5000, len(new_obs)):
	out = model.observe(new_obs[j].reshape(1, new_obs[j].shape[0]))
	if (j >= 5055+100):
		model.t_incipient = 5055+100
		model.initialize_model()	
		history = np.empty((1, 1))	
		n = 0
		o = 0
		while True:
			if (n >= 2):
				next_steps = next_steps + 10
				plt.axvline(140, color='b', linestyle='dashed')
				plt.axvline(165, color='#ff0000', linestyle='dashed')
				plt.axvline(100, color='#FFA500', linestyle='dashed')

			Yp, _ = model.predict(next_steps)
			model.observe(new_obs[j:j+next_steps])
			model.update()
			l = len(Yp)
			# plt.plot(np.array([k for k in range(100, 100+l-next_steps)]), Yp.squeeze()[:l-next_steps]/10.0)
			plt.plot(np.array([k for k in range(100+l-next_steps, 100+l)]), Yp.squeeze()[l-next_steps:l]/10.0)
			plt.plot(model.HI_raw[model.t_incipient-100:model.t_incipient+model.t_current]/10.0, color='g', label='HI')
			plt.plot(model.HI[model.t_incipient-100:model.t_incipient+model.t_current]/10.0, color='k', label='HI Filtered')
			# plt.scatter(np.array([k for k in range(model.t_current)]).reshape(len(model.HI[model.t_incipient:model.t_incipient+model.t_current]), 1), model.HI[model.t_incipient:model.t_incipient+model.t_current], label='observations', color='k', marker='o', alpha=0.3)
			x = np.array([(i) for i in range(len(model.HI[model.t_incipient-100:model.t_incipient+model.t_current]))])

			plt.plot(x, [1.0 for i in range(len(model.HI[model.t_incipient-100:model.t_incipient+model.t_current]))], linestyle='dashed', color='#ff0000')

			plt.plot(x, [0.1907940 for i in range(len(model.HI[model.t_incipient-100:model.t_incipient+model.t_current]))], linestyle='dashed', color='#FFA500')
			ax = plt.gca()
			ax.set_facecolor((0.827, 0.827, 0.827))
			plt.text(1, 1.01, 'Alarm', color='#FF0000')
			plt.text(1, 0.2079, 'Warning', color='#FFA500')
			plt.legend()
			plt.title('Remaining Useful Life (RUL) of Bearing')
			plt.xlabel('timesteps')
			plt.ylabel('Health')
			plt.legend()
			plt.show()
			j += next_steps
			n += 1
			o += next_steps
			print (n, o)
# # 		