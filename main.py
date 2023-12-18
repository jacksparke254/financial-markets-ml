# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:07:08 2023

@author: jacks
"""

#%%

from functions import model_pred

import yfinance as yf
import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.backend import clear_session
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer, accuracy_score, classification_report, confusion_matrix, mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import GRU
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
import time
import datetime
import keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
#from keras.optimizers import SGD


dataset_name = 'EURUSD_60d30m'

dataset_name = 'EURUSD_1'


#%%
dataset_name = 'EURUSD_15m'
model = model_pred()



x_train, y_train, x_test, y_test, scaler = model.importer(symbol = 'EURUSD=X', period='60d', interval='15m', lookback = 30)


#x_train, y_train, x_test, y_test, scaler = model.importer_aug(symbol = 'EURUSD=X', period='60d', interval='15m', lookback = 30)

#x_train, y_train, x_test, y_test = model.oanda_importer_pct(symbol = 'EUR_USD', interval='M15', lookback = 30)

#x_train, y_train, x_test, y_test, scaler = model.split_df(lookback=45, filepath = filepath)

history, regressor = model.RNN(dataset_name = dataset_name, units1=2048, units2= 2048,units3 = 0, epochs= 100, model = "LSTM")
    
history, RNN_pred, counter = model.RNN_test(dataset_name, "LSTM")


#%%

