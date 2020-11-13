# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:43:01 2020

@author: IPTLP0018
Updated By: IPTLP0032

This block provides the various metrics to identify the suitable Health indicators.
Given a numpy matrix with each column representing some degrading features, this block provides the scores for 
each features in relation to degradation

score = computeHIScore(input, method_name = '')

input should be a numpy array with finite numeric values.
methods_name is a string with:
mon - monotonicity
pos_mon - positive monotonicity
neg_mon - negative monotonocity

This function returns the score, the higher the suitable that features are to represent the
degradation characteristics. The score value will be between 0 and 1.
"""

from IntelliMaint.data_analysis import SOM
from IntelliMaint import Utils
import numpy as np
import pandas as pd

class HealthIndicator:

	def __init__(self, chunk_size=None, order=1):
		'''
		Args
		----
		X -- Input features (prospective health indicators)
		algo --  Algorithms to check the preferred feature as health indicator
		chunk_size -- filtering window
		order -- order of the median filter

		Returns
		-------
			Nothing.
		'''
		self.chunk_size = chunk_size
		self.order = order

	def monotonicity(self, data):
		'''
		Monotonicity characterizes an increasing or decreasing trend. 
		It can be measured by the absolute difference of “positive” and “negative” derivatives for each feature.

		    Javed, Kamran, et al. "Enabling health monitoring approach based on vibration data for accurate prognostics." 
		    IEEE Transactions on Industrial Electronics 62.1 (2014): 647-656.

		Monotonicity will only the metrics with time
		'''
		utils = Utils()
		temp = np.diff(data, axis=0)
		score = np.abs(utils.count_pos(temp)-utils.count_neg(temp))/(data.shape[0]-1)
		return score

	def pos_monotonicity(self, data):
		'''
		Positive Prognosability: Characterizes an increasing trend. 
		Liao, Linxia, Wenjing Jin, and Radu Pavel. 
		"Enhanced restricted Boltzmann machine with prognosability regularization for prognostics and health assessment." 
		IEEE Transactions on Industrial Electronics 63.11 (2016): 7076-7083.
		'''
		utils = Utils()
		temp1 = np.diff(data, axis=0)
		temp2 = np.diff(data, 2, axis=0)
		score = (utils.count_pos(temp1)/(data.shape[0]-1) + utils.count_pos(temp2)/(data.shape[0]-2))/2
		return score

	    
	def neg_monotonicity(self, data):
		'''
		Negative Prognosability: Characterizes an decreasing trend. 
		Liao, Linxia, Wenjing Jin, and Radu Pavel. 
		"Enhanced restricted Boltzmann machine with prognosability regularization for prognostics and health assessment." 
		IEEE Transactions on Industrial Electronics 63.11 (2016): 7076-7083.
		'''
		utils = Utils()
		temp1 = np.diff(data, axis=0)
		temp2 = np.diff(data,2,axis=0)
		score = (utils.count_neg(temp1)/(data.shape[0]-1) + utils.count_neg(temp2)/(data.shape[0]-2))/2
		return score

	def linear_trendability(self, data):
	    t = np.linspace(0,len(data)-1,num=len(data))
	    if(len(data.shape) == 1):
	        return  np.asarray(pearsonr(t,data)[0])
	    t = np.linspace(0,len(data)-1,num=len(data))
	    return np.asarray([pearsonr(t,column)[0] for column in data.T])

	def nonlinear_trendability(self, data):
	    t = np.linspace(0,len(data)-1,num=len(data))
	    if(len(data.shape) == 1):
	        return np.asarray(spearmanr(t,data)[0])
	    return np.asarray([spearmanr(t,column)[0] for column in data.T])

	def computeScore(self, data, method='mon'):
	    # check1(data)
	    
	    if(method == 'mon'):
	        score = self.monotonicity(data)
	    elif(method == 'pos_mon'):
	        score = self.pos_monotonicity(data)
	    elif(method == 'neg_mon'):
	        score = self.neg_monotonicity(data)
	    elif(method == 'lin_trend'):
	        score = self.linear_trendability(data)
	    elif(method == 'nonlin_trend'):
	        score = self.nonlinear_trendability(data)
	    else:
	        print('Enter a valid option {\'mon\',\'pos_mon\',\'neg_mon\'}')
	    return score

	def computeHIScore(self, data, method='mon'):
	    if(self.chunk_size == None):
	        return self.computeScore(data, method)
	    if(self.order%2 == 0):
	        self.order = self.order + 1
	        data = signal.medfilt(data, self.order)
	    score = []
	    for i in range(0,len(data), self.chunk_size):
	        score.append(self.computeScore(input_data[i:i+self.chunk_size], method_name=method))
	    return np.asarray(score).reshape(1, len(score))

