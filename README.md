# IntelliMaint - Diagnostics & Prognostics Library

A python based library to estimate future health state and predict any degradations in the subsystem/components. 

This is a pre-alpha version 0.1.

### Input :  
Features that characterise the subsystem/component 

### Process:  

Feature Extraction, Feature fusion and Gaussian Process Regression (GPR)

### Output :  

The output is the Remaining Useful Life (RUL) estimate of the subsystem/component.

## Installing the package:
- Clone the repository
```
$ git clone https://github.com/intellipredikt/intellimaint.git
$ cd intellimaint
```
- Install the package using python
```
$ python setup.py install
```

## Usage

### 1. Implement Health Indicator Class
```
class FeatureIndicator:
	def __init__(self):
		pass

	def health_indicator(self, X):
		return X
```
### 2. Implement Degradation Model Class
```
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

```

### 3. Sample Usage
```
from IntelliMaint.rul_models import DegradationModel
from IntelliMaint.health_indicators import FeatureIndicator
from IntelliMaint import Utils

utils = Utils()

# step 1: Get Features
soh = utils.get_features('battery') # state of health

Y = soh
X = np.array([i for i in range(len(soh))]).reshape(len(soh), 1)

train_X = X[:100]
train_y = Y[:100]

test_X = X
test_y = Y

# step 2: # Use initial 100 obs to prepare model
model = BatteryRULModel(train_y, train_y) 
model.t_incipient = 0
model.t_current = 99
model.initialize_model()

# step 3: # Predict, Observe and Update
Yp, Vp = model.predict()
model.observe(Y[100:100+(i+1)*20])
model.update()
```
## Result
![Sample Result](images/RUL_Battery_Animation.gif "RUL Prediction" )

## How to use the library for health prediction of other components?

* Load data
* Extract Features
* Train Feature Fusion for Health Indicator (HI) or,
* Use the degrading feature for HI (as in the case of battery).
* Observe the HI and trigger the prediction at the incipient fault time.
* Initialize the degradation model with initial few observations after the incipient fault time.
* Train the degradation model. 
* Predict mult-step in the future, observe the true HI values.
* Update the degradation model.

## Reference

[Hong, Sheng, Zheng Zhou, Chen Lu, Baoqing Wang, and Tingdi Zhao. "Bearing remaining life prediction using Gaussian process regression with composite kernel functions." *Journal of Vibroengineering 17* , no. 2 (2015): 695-704.]

## For any queries 
iptgithub@intellipredikt.com
