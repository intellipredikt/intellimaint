# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:14:14 2020

@author: Ajay
Updated By: IPTLP0032

"""
#%%-----------------Section1: Importing fuction -------------------------------#
# from IPython import get_ipython
# get_ipython().magic('reset -sf')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os as os
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.externals import joblib
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Input, LSTM, Dropout
from keras.layers import RepeatVector,TimeDistributed
from keras import optimizers
# from tcn import TCN
import mplcursors
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import seaborn as sns
import tensorflow as tf
from minisom import MiniSom
import scipy.signal

sns.set(style="whitegrid")
warnings.filterwarnings("ignore")
plt.close('all')
np.random.seed(203)


#%%------------Section-2: Important Function for conversion and prediction----#
# creating 3D input matrix with lookback(no. of rows to look back) 

global anomaly_count,count

class AutoEncoder:
	def __init__(self):
		pass

	def train(self, x,L1=100,L2=100,e_dim=2,a_func='relu',b_size=30,epochs=100):
	    """
	    Autoencoder training with 2 encoder and 2 decoder dense layers. \n
	    Input Parameters
	    ----------------
	    x,L1,L2,e_dim,a_func,b_size,epochs:
	        x : Raw input data \n
	        L1 : Dense layer 1 with neurons (default=100) \n
	        L2 : Dense layer 2 with neurons (default=100) \n
	        e_dim : Latent representation or encoding dimension (default=2)
	        a_func : Activation function (default=relu) \n
	        b_size : Batch size (default=30)
	        epochs : Number of interations (default =100)
	    Returns
	    -------
	    AE,model,scaler:
	        AE : Autoencoder history from the trained model \n
	        model : Autoencoder model \n
	        scaler : Scaler used for scaling input data
	    """ 
	    x_train,x_test=train_test_split(x,test_size=0.2)
	    scaler=preprocessing.StandardScaler().fit(x_train)
	    x_train=scaler.transform(x_train)
	    x_test=scaler.transform(x_test)
	    ncol = x_train.shape[1]
	    input_dim = Input(shape = (ncol, ))

	    # DEFINE THE ENCODER LAYERS
	    encoded1 = Dense(L1, activation = 'linear')(input_dim)
	    encoded2 = Dense(L2, activation = a_func)(encoded1)
	    encoded3 = Dense(e_dim, activation = a_func)(encoded2)

	    # DEFINE THE DECODER LAYERS
	    decoded1 = Dense(L2, activation = a_func)(encoded3)
	    decoded2 = Dense(L1, activation = a_func)(decoded1) 
	    decoded3 = Dense(ncol, activation = 'linear')(decoded2)

	    model = Model(inputs = input_dim, outputs = decoded3)
	    model.compile(optimizer = 'SGD', loss = 'mse',metrics=['accuracy'])
	    AE = model.fit(x_train, x_train, epochs = epochs, batch_size = b_size,
	                         shuffle = True,validation_data = (x_test, x_test)) 
	    training_loss = AE.history['loss']
	    test_loss = AE.history['val_loss']
	    plt.figure()
	    plt.plot(training_loss, 'r--')
	    plt.plot(test_loss, 'b-')
	    plt.legend(['Training Loss', 'Test Loss'])
	    plt.xlabel('Epoch')
	    plt.ylabel('Loss')
	    plt.title('encoding_dim=' + str(e_dim))
	    return AE,model,scaler

	def predict(self, model,scaler,data):
	    """
	    Predicting target data from autoencoder model. \n
	    Input Parameters
	    ----------------
	    model,scaler,data:
	        model : Pre trainedAutoencoder model\n
	        scaler : Scaler used for scaling input data \n
	        data : Data from csv/xlsx file \n
	    Returns
	    -------
	    RE:
	        RE : Reconstruction error from the target data \n
	    """
	    x=scaler.transform(data)
	    pred=model.predict(x)
	    RE=np.abs(pred-x)
	    return RE  

	def plot(self, model,scaler,data,c_n,title):
	    """
	    Autoencoder model output plot . \n
	    Input Parameters
	    ----------------
	    model,c_n,data,title:
	        model : Autoencoder model \n
	        scaler : Scaler used for scaling input data \n
	        data: Input data should be of dataframe (used as index in plot) \n
	        c_n : Column number \n
	        title : Title of plot \n
	    Returns
	    -------
	    qe:
	        qe : Computed quantisation error \n
	    """
	    plt.figure()
	    x=scaler.transform(data)
	    pred=model.predict(x)
	    ax1=plt.subplot(2,1,1)
	    plt.plot(data.index,x[:,c_n])
	    plt.plot(data.index,pred[:,c_n])
	    plt.legend(['Data','Reconstructed'])
	    plt.title(title,fontsize=16,fontweight='bold')
	    plt.xticks(rotation=0, ha='right')
	    date_form = DateFormatter("%d/%b/%y\n%H:%M") # date and year
	    ax1.xaxis.set_major_formatter(date_form)  
	    ax2=plt.subplot(2,1,2,sharex=ax1)
	    RE=np.abs(pred[:,c_n]-x[:,c_n])
	    plt.plot(data.index,RE,'k')
	    plt.title('Reconstruction error',fontsize=16,fontweight='bold')
	    plt.xticks(rotation=0, ha='right')
	    ax2.xaxis.set_major_formatter(date_form)
	    mplcursors.cursor()
	    plt.tight_layout()

class SOM:
	def __init__(self):
		pass

	# SOM Algorithm
	def train(self, x,w1=50,w2=50,sigma=0.1,lr=0.5,n_iter=500):
	    """
	    Training Self Organising map for input dataset\n
	    Input Parameters
	    ----------------
	    x,w1,w2,sigma,l_rate,n_int:
	        x : Input data \n
	        w1 & w2 : Dimension of output window \n
	        sigma : Radius
	        lr : Learning rate
	        n_iter : Number of interations
	    Returns
	    -------
	    som,q_error,scaler:
	        som: self organinsing map model \n
	        q_error: quantisation error of predicted data\n
	        scaler: scaling input data \n
	    """
	    scaler = preprocessing.MinMaxScaler()
	    x = scaler.fit_transform(x)
	    som = MiniSom(w1, w2, x.shape[1], sigma=sigma, learning_rate=lr)
	    som.random_weights_init(x)
	    som.train_random(x, n_iter) 
	 
	    return som, scaler

	def predict(self, som,data,scaler):
	    """
	    Prediction from SOM model \n
	    Input Parameters
	    ----------------
	    som,data,scaler:
	        som : SOM trained model \n
	        data : Input data from file \n
	        scaler : Scaler used for scaling data
	    Returns
	    -------
	    q_error:
	        q_error : Quantisation error from SOM \n
	    """
	    x=scaler.transform(data)
	    error=som.quantization(x)
	    q_error=self.quant_error(x,error)
	    return q_error
	    
	def quant_error(self, x,qant):
	    """
	    Quantisation error calculated based on input and predicted data. \n
	    Input Parameters
	    ----------------
	    x,qant:
	        x : Input data \n
	        qant : predicted data from som model
	    Returns
	    -------
	    qe:
	        qe : Computed quantisation error \n
	    """
	    qe = [] # quantization error
	    for i in range(len(x)):
	        qe.append(np.linalg.norm(qant[i] - x[i])) 
	    return np.array(qe)
