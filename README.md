# IntelliMaint - Diagnostics & Prognostics Library

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

### For any queries 
iptgithub@intellipredikt.com
