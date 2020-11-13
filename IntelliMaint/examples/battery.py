import numpy as np
from IntelliMaint.rul_models import GPRDegradationModel
from IntelliMaint.health_indicators import FeatureIndicator
import matplotlib.pyplot as plt
from IntelliMaint import Utils
import imp
import os
import csv

# file_name = 'test.csv'
# file = open(file_name, 'w')
# writer = csv.writer(file)
# writer.writerow(['input (Y)', 'output (Yp)', 'confidence (Vp)', 'status', 'action', 'rul'])
# dirname = os.path.dirname(__file__)
# hi_class_file = dirname + "/__pycache__/health_indicators.cpython-36.pyc"
# my_module = imp.load_compiled("my_module", hi_class_file)

class BatteryRULModel(FeatureIndicator, GPRDegradationModel):
	def __init__(self, obs, obs1):
		FeatureIndicator.__init__(self)
		HI = self.health_indicator(obs)
		self.HI = HI
		self.obs = obs
		self.HI_to_train_model = None
		self.t_incipient = 0 # incipient fault time
		self.t_current = 0

	def observe(self, X): # new raw observation
		self.obs = np.concatenate((self.obs, X), axis=0)
		self.HI = self.health_indicator(self.obs) # new observations
		self.t_current += len(X)
		return self.HI

	def predict(self):
		X = np.array([i for i in range(167)]).reshape(167, 1)
		Yp, Vp = GPRDegradationModel.predict(self, X)
		return Yp, Vp

	def initialize_model(self):
		GPRDegradationModel.__init__(self, self.HI)

	def update(self):
		l = self.t_current
		X = np.array([i for i in range(self.t_current)]).reshape(l, 1)
		GPRDegradationModel.update(self, X, self.HI[self.t_incipient:self.t_incipient+self.t_current])
		

class RULModelDemo:
	def __init__(self):
		utils = Utils() # i-th data-set out of 218 data-sets
		# step 1: Get Features
		soh = utils.get_features('battery')

		Y = soh
		X = np.array([i for i in range(len(soh))]).reshape(len(soh), 1)

		# initial observation
		obs = 1
		train_X = X[:obs]
		train_y = Y[:obs]

		test_X = X
		test_y = Y

		self.model = BatteryRULModel(train_y, train_y) # use initial 100 obs to prepare model
		self.model.t_incipient = 0
		self.model.t_current = obs - 1
		self.model.initialize_model()
		self.i = 0
	def run(self, new_obs):
		self.i += 1
		Yp, Vp = self.model.predict()
		alarm_index, = np.where(Yp.squeeze() <= 70)
		warning_index, = np.where(Yp.squeeze() <= 75)
		rul_ready = False
		warning_ready = False

		action = 'NO ACTION'
		status = 'NORMAL'
		rul = '--'
		if (len(alarm_index) > 0):
			alarm_index = alarm_index[0] # the first value
			# print (Vp.squeeze()[alarm_index])
			rul = (168 - self.i) * 0.20 # days
			rul = round(rul, 2)
			rul_ready = True
		if (len(warning_index) > 0):
			if (self.i < alarm_index and self.i >= warning_index[0]):
				warning_index = warning_index[0]
				status = 'WARNING'
				action = 'INSPECT'
				rul = (168 - self.i) * 0.20 # days
				rul = round(rul, 2)
		
		if (rul_ready and self.i >= alarm_index):
			status = 'ALARM'
			action = 'REPLACE'

		self.model.observe(new_obs.reshape(1, 1))
		self.model.update()

		return Yp, Vp, status, action, rul

def date_diff_in_Seconds(dt2, dt1):
  timedelta = dt2 - dt1
  return timedelta.days * 24 * 3600 + timedelta.seconds

utils = Utils() # i-th data-set out of 218 data-sets
# step 1: Get Features
soh = utils.get_features('battery')[1:]

# updated_times = []
# for t in times:
# 	time_string = (str(t.squeeze()))
# 	t_ = time_string.split(',', 2)
# 	date_string = t_[0]
# 	time_string = t_[1]

# 	day, month, year = date_string.split(' ', 3)
# 	h, m, s = time_string.split(':', 3)

# 	if (month == 'Apr'):
# 		month = '04'
# 	if (month == 'May'):
# 		month = '05'

# 	temp = year+'-'+month+'-'+day+' '+h+':'+m+':'+s
# 	temp = datetime.strptime(temp, '%Y-%m-%d %H:%M:%S')
# 	updated_times.append(temp)

# diff_between_times = []

# for i in range(1, len(updated_times)):
# 	diff = (date_diff_in_Seconds(updated_times[i], updated_times[i-1]) / 3600.0) / 24
# 	diff_between_times.append(diff)


Y = soh
X = np.array([i for i in range(len(soh))]).reshape(len(soh), 1) # days

model = RULModelDemo()

test_X = X
test_y = Y

# total 167 new observation will be streamed
for i in range(167):
	# plt.scatter(np.array([i for i in range(len(test_y[:i]))]), test_y[:i], label='Health Indicator', color='k', marker='x')
	plt.plot(model.model.HI[:i], color='k', label='Health Indicator')
	plt.plot([75 for i in range(len(soh))], linestyle='dashed', color='#FF8C00')
	plt.plot([70 for i in range(len(soh))], linestyle='dashed', color='#ff0000')
	
	Yp, Vp, status, action, rul = model.run(Y[i]) # Y[i] is the observation/input shape numpy(1, 1)
	# writer.writerow([Y[i], Yp.squeeze(), Vp.squeeze(), status, action, rul])
	# Vp = np.sqrt(Vp.squeeze())

	# if (i == 33):
	# 	plt.plot(X[i:], Yp.squeeze()[i:], label='prediction', color='k')
	# 	plt.fill_between(test_X[i:].reshape(1, 167-i).squeeze(), Yp.squeeze()[i:], Yp.squeeze()[i:] + Vp[i:], color='k', alpha=.5)
	# 	plt.fill_between(test_X[i:].reshape(1, 167-i).squeeze(), Yp.squeeze()[i:], Yp.squeeze()[i:] - Vp[i:], color='k', alpha=.5)
	# 	plt.axvline(i, color='k', linestyle='dashed')

	# if (i == 50):
	# 	plt.plot(X[i:], Yp.squeeze()[i:], label='prediction', color='b')
	# 	plt.fill_between(test_X[i:].reshape(1, 167-i).squeeze(), Yp.squeeze()[i:], Yp.squeeze()[i:] + Vp[i:], color='b', alpha=.5)
	# 	plt.fill_between(test_X[i:].reshape(1, 167-i).squeeze(), Yp.squeeze()[i:], Yp.squeeze()[i:] - Vp[i:], color='b', alpha=.5)
	# 	plt.axvline(i, color='b', linestyle='dashed')

	# if (i == 92):
	# 	plt.plot(X[i:], Yp.squeeze()[i:], label='prediction', color='g')
	# 	plt.fill_between(test_X[i:].reshape(1, 167-i).squeeze(), Yp.squeeze()[i:], Yp.squeeze()[i:] + Vp[i:], color='g', alpha=.5)
	# 	plt.fill_between(test_X[i:].reshape(1, 167-i).squeeze(), Yp.squeeze()[i:], Yp.squeeze()[i:] - Vp[i:], color='g', alpha=.5)
	# 	plt.axvline(i, color='g', linestyle='dashed')

	if (i >= 2):
		if (Vp.squeeze()[i] <= 2.70 and (status == 'WARNING')):
			plt.plot(X[i:], Yp.squeeze()[i:], label='Prediction', color='#FF8C00')
			plt.title('STATUS : ' +status+ ' | ACTION : '+action+' | RUL : '+str(rul)+ ' days')
			plt.axvline(i, color='#FF8C00', linestyle='dashed')
			plt.fill_between(test_X[i:].reshape(1, 167-i).squeeze(), Yp.squeeze()[i:], Yp.squeeze()[i:] + Vp[i:], color='#FF8C00', alpha=.2, label='Warning confidence')
			plt.fill_between(test_X[i:].reshape(1, 167-i).squeeze(), Yp.squeeze()[i:], Yp.squeeze()[i:] - Vp[i:], color='#FF8C00', alpha=.2)
		elif ((status == 'ALARM')):
			plt.plot(X[i:], Yp.squeeze()[i:], label='Prediction', color='r')
			plt.title('STATUS : ' +status+ ' | ACTION : '+action+' | RUL : '+str(rul)+ ' days')
			plt.axvline(i, color='r', linestyle='dashed')
			plt.fill_between(test_X[i:].reshape(1, 167-i).squeeze(), Yp.squeeze()[i:], Yp.squeeze()[i:] + Vp[i:], color='r',  alpha=.2, label='Alarm confidence')
			plt.fill_between(test_X[i:].reshape(1, 167-i).squeeze(), Yp.squeeze()[i:], Yp.squeeze()[i:] - Vp[i:], color='r', alpha=.2)
		else:
			plt.title('STATUS : ' +status+ ' | ACTION : '+action)
			
	plt.xlabel('timesteps (Cycles)')
	plt.ylabel('Health Indicator (Capacity (%))')
	# plt.ylim((-10,100))
	plt.legend(loc="upper right", bbox_transform=plt.gcf().transFigure)
	ax = plt.gca()
	plt.style.use('ggplot')
	ax.set_facecolor((0.827, 0.827, 0.827))
	plt.text(1.01, 75, 'Warning', color='#FFA500')
	plt.text(1.01, 70, 'Alarm', color='#FF0000')
	# plt.tight_layout()

	# plt.show()
	if (i <= 150):
		# break
		# print (samples[i]) # samples[i], Yp[i] (mu), Vp[i] (sigma)
		# mu = Yp[i]
		# sigma = Vp[i]
		# samples = samples[i].reshape(samples[i].shape[1], 1) # 100 samples
		

		# # An "interface" to matplotlib.axes.Axes.hist() method
		# n, bins, patches = plt.hist(x=samples, bins='auto', color='#0504aa',
		#                             alpha=0.7, rwidth=1)
		# plt.grid(axis='y', alpha=0.75)
		# plt.xlabel('Value')
		# plt.ylabel('Frequency')
		# plt.title('My Very Own Histogram')
		# maxfreq = n.max()
		# Set a clean upper y-axis limit.
		# plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

		# plt.savefig('normal_curve.png', dpi=72, bbox_inches='tight')
		# plt.show()
	
	# if (i == 26):
	# 	plt.plot(X[i:], Yp.squeeze()[i:], label='prediction')
	# 	plt.fill_between(test_X[i:].reshape(1, 167-i).squeeze(), Yp.squeeze()[i:], Yp.squeeze()[i:] + Vp[i:], color='g', alpha=.5)
	# 	plt.fill_between(test_X[i:].reshape(1, 167-i).squeeze(), Yp.squeeze()[i:], Yp.squeeze()[i:] - Vp[i:], color='g', alpha=.5)
	# 	plt.axvline(i, color='g', linestyle='dashed')

	# if (i == 91):
	# 	plt.plot(X[i:], Yp.squeeze()[i:], label='prediction')
	# 	plt.fill_between(test_X[i:].reshape(1, 167-i).squeeze(), Yp.squeeze()[i:], Yp.squeeze()[i:] + Vp[i:], color='b', alpha=.5)
	# 	plt.fill_between(test_X[i:].reshape(1, 167-i).squeeze(), Yp.squeeze()[i:], Yp.squeeze()[i:] - Vp[i:], color='b', alpha=.5)
	# 	plt.axvline(i, color='b', linestyle='dashed')

	# if (i == 132):
	# 	plt.plot(X[i:], Yp.squeeze()[i:], label='prediction')
	# 	plt.fill_between(test_X[i:].reshape(1, 167-i).squeeze(), Yp.squeeze()[i:], Yp.squeeze()[i:] + Vp[i:], color='r', alpha=.5)
	# 	plt.fill_between(test_X[i:].reshape(1, 167-i).squeeze(), Yp.squeeze()[i:], Yp.squeeze()[i:] - Vp[i:], color='r', alpha=.5)
		# plt.axvline(i, color='r', linestyle='dashed')



		plt.savefig(str(i)+'.png')
		plt.clf()
file.close()