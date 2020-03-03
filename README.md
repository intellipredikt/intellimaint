# IntelliMaint - Diagnostics & Prognostics Library

IntelliMaint library offers an integrated set of functions written in Python that aids in vehicle health diagnostics. The library includes various state-of-the-art algorithms that are accessible by users with simple APIs. Thus the library accelerates the analysis and the development phase for even the novice users. Most of the APIs are provided with intuitive visualizations that guide in choosing the apt algorithm for a system component. Each block in the library is built as an independent module thus used as a plug and play module. The expert users are provided with a provision to add new algorithms for specific use cases. 

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


## Result
![Sample Result](images/RUL_Battery_Animation_2.gif "RUL Prediction" )

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

Hong, Sheng, Zheng Zhou, Chen Lu, Baoqing Wang, and Tingdi Zhao. "Bearing remaining life prediction using Gaussian process regression with composite kernel functions." *Journal of Vibroengineering 17* , no. 2 (2015): 695-704.

## For any queries 
iptgithub@intellipredikt.com
