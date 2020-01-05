import numpy as np
from IntelliMaint.rul_models import DegradationModel
from IntelliMaint.health_indicators import FeatureIndicator
import matplotlib.pyplot as plt
from IntelliMaint import Utils

class BatteryRULModel(FeatureIndicator, DegradationModel):
	def __init__(self, obs, obs1):
		FeatureIndicator.__init__(self)
		HI = self.health_indicator(obs)
		self.HI = HI
		self.obs = obs
		self.HI_to_train_model = None
		self.t_incipient = 0 # incipient fault time
		self.t_current = 99

	def observe(self, X): # new raw observation
		self.obs = np.concatenate((self.obs, X), axis=0)
		self.HI = self.health_indicator(self.obs) # new observations
		self.t_current += len(X)
		return self.HI

	def predict(self):
		X = np.array([i for i in range(168)]).reshape(168, 1)
		Yp, Vp = DegradationModel.predict(self, X)
		return Yp, Vp

	def initialize_model(self):
		DegradationModel.__init__(self, self.HI)

	def update(self):
		l = self.t_current
		X = np.array([i for i in range(self.t_current)]).reshape(l, 1)
		DegradationModel.update(self, X, self.HI[self.t_incipient:self.t_incipient+self.t_current])
		

utils = Utils() # i-th data-set out of 218 data-sets

# step 1: Get Features
soh = utils.get_features('battery')

Y = soh
X = np.array([i for i in range(len(soh))]).reshape(len(soh), 1)

train_X = X[:100]
train_y = Y[:100]

test_X = X
test_y = Y

model = BatteryRULModel(train_y, train_y) # use initial 100 obs to prepare model
model.t_incipient = 0
model.t_current = 99
model.initialize_model()

colors_ = ['g', 'b', '#FFA500', 'r']

for i in range(4):
	plt.plot(test_y, label='HI', color='k')
	plt.plot(np.array([100 for i in range(len(soh))]), np.array([i for i in range(len(soh))]), linestyle='dashed', color='g')
	plt.plot(np.array([120 for i in range(len(soh))]), np.array([i for i in range(len(soh))]), linestyle='dashed', color='b')
	plt.plot(np.array([140 for i in range(len(soh))]), np.array([i for i in range(len(soh))]), linestyle='dashed', color='#FFA500')
	plt.plot(np.array([160 for i in range(len(soh))]), np.array([i for i in range(len(soh))]), linestyle='dashed', color='r')

	plt.text(100, 100, 'Prediction at \n 100')
	plt.text(120, 100, 'Prediction at \n 120')
	plt.text(140, 100, 'Prediction at\n140\nWarning!')
	plt.text(160, 100, 'Prediction at\n160\nAlert!')

	plt.plot([75 for i in range(len(soh))], linestyle='dashed', color='#FFA500')
	plt.plot([70 for i in range(len(soh))], linestyle='dashed', color='#ff0000')

	Yp, Vp = model.predict()

	plt.plot(X[:100+(i)*20], Yp.squeeze()[:100+(i)*20])
	# print(100+(i)*20)
	# plt.plot([i for i in range(100+(i+1)*20, 100+(i+1)*20+len(Yp.squeeze()[100+(i+1)*20:]))], Yp.squeeze()[100+(i+1)*20:], label='pred')
	plt.fill_between(test_X.reshape(1, 168).squeeze(), Yp.squeeze(), Yp.squeeze() + Vp.squeeze(), color=colors_[i], alpha=.5)
	plt.fill_between(test_X.reshape(1, 168).squeeze(), Yp.squeeze(), Yp.squeeze() - Vp.squeeze(), color=colors_[i], alpha=.5)
	
	model.observe(Y[100:100+(i+1)*20])
	model.update()

	plt.title('RUL for B0005 using GPR')
	plt.xlabel('timesteps (cycles)')
	plt.ylabel('Health (Capacity)')
	# plt.legend()
	ax = plt.gca()
	ax.set_facecolor((0.827, 0.827, 0.827))
	plt.text(1.01, 75, 'Warning', color='#FFA500')
	plt.text(1.01, 70, 'Alarm', color='#ff0000')
	plt.show()
	plt.clf()