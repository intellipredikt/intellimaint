# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:23:23 2020

@author: IPTLP0018
@author: anup
"""

import numpy as np
import matplotlib.pyplot as plt
from grand import IndividualAnomalyInductive, IndividualAnomalyTransductive, GroupAnomaly
import pandas as pd

class AnomalyDetection:

    def __init__(self):
        pass

    def deviation_detection(self, data, mu, sigma, l1 = 4, l2 = 8, l3 = 12): 
        z_s = self.zscore(data,mu, sigma)
        if(len(z_s.shape)>1):
            z_s = z_s[:,0]
        t = np.linspace(0,len(z_s)-1,len(z_s))
        thres1 = l1*sigma
        thres2 = l2*sigma
        thres3 = l3*sigma
        plt.scatter(t[np.where(z_s<=thres1)], z_s[np.where(z_s<=thres1)], color='y', label='Normal', alpha=0.3, edgecolors='none')
        plt.scatter(t[np.where((z_s>thres1) & (z_s<=thres2))], z_s[np.where((z_s>thres1) & (z_s<=thres2))], color='b', label='L1 Threshold', alpha=0.3, edgecolors='none')
        plt.scatter(t[np.where((z_s>thres2) & (z_s<=thres3))], z_s[np.where((z_s>thres2) & (z_s<=thres3))], color='g', label='L2 Threshold', alpha=0.3, edgecolors='none')
        plt.scatter(t[np.where(z_s>thres3)], z_s[np.where(z_s>thres3)], color='r', label='Anomalous points', alpha=0.3, edgecolors='none')
        plt.xlabel('Observation Signal (in samples)')
        plt.ylabel('Anomaly Score')
        plt.title('Anomaly Score Estimation')
        plt.legend()
        return z_s, sigma

    def train_cosmo(self,data, w_martingale = 15,non_conformity = "median",k = 20):
        
        df = data
        
        self.model = IndividualAnomalyInductive(
            w_martingale = w_martingale,
            non_conformity = non_conformity,
            k = k)

        # Fit the model to a fixed subset of the data
        X_fit = data.to_numpy()
        self.model.fit(X_fit)


    def test_cosmo(self, data):
        cols = ['Strangeness', 'P-Values', 'Deviation']
        lst_dict = []
        df = data
        for t, x in zip(df.index, df.values):
            info = self.model.predict(t, x)
    
            lst_dict.append({'Strangeness': info.strangeness,
                             'P-Values':info.pvalue,
                             'Deviation':info.deviation})
            
        # Plot strangeness and deviation level over time
        # gr = model.plot_deviations(figsize=(2000,2000))

        df1 = pd.DataFrame(lst_dict, columns=cols)
        
        return df1['Strangeness'].to_numpy(), df1['P-Values'].to_numpy()
    
    def nonstationary_AD_cosmo(self,data,n,w_martingale,k,non_conformity = "median", ref_group=["hour-of-day"]):
        
        df = self.data
        cols = ['Strangeness', 'P-Values', 'Deviation']
        lst_dict = []
        
        model = IndividualAnomalyTransductive(
                w_martingale = w_martingale,         # Window size for computing the deviation level
                non_conformity = non_conformity,  # Strangeness measure: "median" or "knn"
                k = k,                     # Used if non_conformity is "knn"
                ref_group = ref_group  # Criteria for reference group construction
                )

        
        for t, x in zip(df.index, df.values):
            info = model.predict(t, x)
    
            lst_dict.append({'Strangeness': info.strangeness,
                             'P-Values':info.pvalue,
                             'Deviation':info.deviation})
            
        # Plot strangeness and deviation level over time
        gr = model.plot_deviations(figsize=(2000,2000))

        df1 = pd.DataFrame(lst_dict, columns=cols)
        
        return df1, gr

