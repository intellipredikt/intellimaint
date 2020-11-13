#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 20:00:47 2020

@author: anup
"""

#%% Import Libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
#plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm

#%% Exploratory Analysis Class
class ExploratoryAnalysis:
    def __init__(self,data):
        self.data = data 
    
    def perform_eda(self):
        
        print('------------------------------------')
        print("Performing Exploratory Data Analysis")
        print('------------------------------------')
        print("")
        #print("")
        print ('The train data has {0} rows and {1} columns'.format(self.data.shape[0],self.data.shape[1]))
        print("")
        def basic_stats():
            global basic_stats_df
            print("Descriptive Statistics")
            print('----------------------')
            bs = self.data.describe()
            basic_stats_df = bs
            print(bs)
            #print("")
            #return(bsf)
        
        def missing_value_plot():
            print("Missing Value Analysis")
            print('----------------------')
            k = self.data.columns[self.data.isnull().any()]
            
            
            #visualising missing values
            if len(k) == 0:
                print("Hurray !!! No Missing Values")
            else:
                print("Oops!! Some Values are missing, checkout the plot")
                plt.figure()
                miss = self.data.isnull().sum()/len(self.data)
                miss = miss[miss > 0]
                miss.sort_values(inplace=True)
                miss = miss.to_frame()
                miss.columns = ['count']
                miss.index.names = ['Name']
                miss['Name'] = miss.index
    
                #plot the missing value count
                sns.set(style="whitegrid", color_codes=True)
                sns.barplot(x = 'Name', y = 'count', data=miss)
                plt.xticks(rotation = 90)
                #plt.show()
                
                
        def skewness_calc():
            global skewness_df
            print("")
            print('Skewness Analysis')
            print('-----------------')
            sk = self.data.skew(axis = 0, skipna = True)
            sk = sk.to_frame()
            sk.columns = ["Skewness_Value"]
            skewness_df = sk
            print(sk)
            
        def kurtosis_calc():
            global kurt_df
            print("")
            print('Kurtosis Analysis')
            print('-----------------')
            kr = self.data.skew(axis = 0, skipna = True)
            kr = kr.to_frame()
            kr.columns = ["Kurtosis_Value"]
            kurt_df = kr
            print(kr)


            
        def distribution_check():#assuming last column is the target variable
            #plt.figure()
            num = [f for f in self.data.columns if self.data.dtypes[f] != 'object']
            nd = pd.melt(self.data, value_vars = num)
            sns.set(font_scale=1)

            def dist_meanplot(x, **kwargs):
                sns.distplot(x, hist_kws=dict(alpha=0.3))
                plt.axvline(x.mean(),  color = 'red', label = str("mean: {:.2f}".format(x.mean())))
                plt.axvline(x.median(),linestyle ='--' , color = 'g',label = str("median: {:.2f}".format(x.median())))
                #plt.axvline(x.mode(),  color = 'o', label = str("mode: {:.2f}".format(x.mode())))

            g = sns.FacetGrid(nd, col="variable" ,col_wrap=2, sharex=False, sharey = False)
            g.map(dist_meanplot, "value").set_titles("{col_name}")
            for ax in g.axes.ravel():
                ax.legend(loc='best', fontsize = 'x-small')
            plt.tight_layout()
            plt.show()    

            
        def correlation_check():
            plt.figure()
            corr = self.data.corr()
            mask = np.zeros_like(corr)
            mask[np.triu_indices_from(mask)] = True
            with sns.axes_style("white"):
                ax = sns.heatmap(corr,linewidths=.5, mask=mask, cmap="YlGnBu",annot=True)
                bottom, top = ax.get_ylim()
                ax.set_ylim(bottom + 0.5, top - 0.5)
                #ax.set_xlim(xmin=0.0, xmax=1000)
            
            plt.title('Correlation Graph')
            plt.tight_layout()
            plt.show()
            
            
#        missing_values()
        #return(basic_stats()) 
        basic_stats()
        missing_value_plot()
        skewness_calc()
        kurtosis_calc()
        distribution_check()
        correlation_check()
        return(basic_stats_df,skewness_df,kurt_df)