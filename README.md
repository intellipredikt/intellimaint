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
$ git clone https://github.com/ioarun/intellimaint.git
$ cd intellimaint
```
- Install the package using python
```
$ python setup.py install
```

## Usage
```
from IntelliMaint import BearingPrognostics

b = BearingPrognostics()
b.train_som()
b.train_gpr()
b.predict_rul()
```
## Result
![Sample Result](images/results.png "RUL Prediction" )

## How to use the library for health prediction of other components?

* Load data
* Extract Features
* Train SOM for Feature Fusion
* Find the Minimum Quantization Error (MQE) of each feature.
* Regress over MQE to predict RUL.


### For any queries 
iptgithub@intellipredikt.com
