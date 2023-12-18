# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:11:30 2023

@author: jacks
"""
import sys
sys.path.insert(0,'C:/Users/jacks/Documents/FOREX ML')

import yfinance as yf
import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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
import numpy as np
import keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
import threading
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from oandapyV20 import API
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
#from keras.optimizers import SGD
#%%

class model_pred():
    def split_df(self, lookback, filepath):
        
        header = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = pd.read_csv(filepath, header =None)
        df.columns = header
        df = df[50000:]
        
        train, test = train_test_split(df, train_size = .75, shuffle = False)
        
        train, test = train.filter(["Close"]), test.filter(["Close"])
        
        scaler = MinMaxScaler(feature_range=(0,1))
    
        train_scaled = scaler.fit_transform(train.values)
        lookback = lookback
        x_train = []
        y_train = []
        for i in range(lookback, len(train_scaled)):
            x_train.append(train_scaled[i-lookback:i, 0])
            y_train.append(train_scaled[i, 0])
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        y_test = test.iloc[lookback:, 0:].values
        
        test_inputs = test
        test_inputs = scaler.transform(test_inputs)
        x_test = []
        
        for i in range(lookback, len(test_inputs)):
            x_test.append(test_inputs[i-lookback:i, 0])
        
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        return x_train, y_train, x_test, y_test, scaler 
        
        
    def importer(self, symbol,period,interval,lookback):
        forex_data_minute = yf.download(symbol, period=period, interval=interval)[:-1]
        
        train, test = train_test_split(forex_data_minute, train_size = .75, shuffle = False)
        train, test = train.filter(["Close"]), test.filter(["Close"])
        
        scaler = MinMaxScaler(feature_range=(0,1))
    
        train_scaled = scaler.fit_transform(train.values)
        lookback = lookback
        x_train = []
        y_train = []
        for i in range(lookback, len(train_scaled)):
            x_train.append(train_scaled[i-lookback:i, 0])
            y_train.append(train_scaled[i, 0])
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        y_test = test.iloc[lookback:, 0:].values
        
        test_inputs = test
        test_inputs = scaler.transform(test_inputs)
        x_test = []
        
        for i in range(lookback, len(test_inputs)):
            x_test.append(test_inputs[i-lookback:i, 0])
        
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        return x_train, y_train, x_test, y_test, scaler
    
    def importer_aug(self, symbol,period,interval,lookback):
        forex_data_minute = yf.download(symbol, period=period, interval=interval)[:-1]
        
        train, test = train_test_split(forex_data_minute, train_size = .75, shuffle = False)
        train, test = train.filter(["Close"]), test.filter(["Close"])
        
        mean_train = np.mean(train)
        mean_test = np.mean(test)

        train[train < mean_train] *= 0.99
        test[test < mean_test] *= 0.99
        
        train[train > mean_train] *= 1.01
        test[test > mean_test] *= 1.01
        
        scaler = MinMaxScaler(feature_range=(0,1))
    
        train_scaled = scaler.fit_transform(train.values)
        lookback = lookback
        x_train = []
        y_train = []
        for i in range(lookback, len(train_scaled)):
            x_train.append(train_scaled[i-lookback:i, 0])
            y_train.append(train_scaled[i, 0])
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        y_test = test.iloc[lookback:, 0:].values
        
        test_inputs = test
        test_inputs = scaler.transform(test_inputs)
        x_test = []
        
        for i in range(lookback, len(test_inputs)):
            x_test.append(test_inputs[i-lookback:i, 0])
        
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        return x_train, y_train, x_test, y_test, scaler
    
    def oanda_importer(self,symbol, interval, lookback):
        api_key = 
        accountID = 
        
        
        client = API(access_token=api_key)
        
        params = {
          "count": 5000,
          "granularity": interval ##M15
        }
        
        r = instruments.InstrumentsCandles(instrument=symbol,params=params)
        client.request(r)
        
        
        for i in range(0,4999):
            if i == 0:
                series = pd.DataFrame(columns=['Date', "Close"], index=range(4999))
                series['Date'][i] = pd.to_datetime(r.response['candles'][0]['time'])
                series['Close'][i] = float(r.response['candles'][i]['mid']['c'])
            else:
                series['Date'][i] = pd.to_datetime(r.response['candles'][i]['time'])
                series['Close'][i] = float(r.response['candles'][i]['mid']['c'])

        forex_data_minute = series
        
        train, test = train_test_split(forex_data_minute, train_size = .75, shuffle = False)
        train, test = train.filter(["Close"]), test.filter(["Close"])
        
        scaler = MinMaxScaler(feature_range=(0,1))
    
        train_scaled = scaler.fit_transform(train.values)
        lookback = lookback
        x_train = []
        y_train = []
        for i in range(lookback, len(train_scaled)):
            x_train.append(train_scaled[i-lookback:i, 0])
            y_train.append(train_scaled[i, 0])
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        y_test = test.iloc[lookback:, 0:].values
        
        test_inputs = test
        test_inputs = scaler.transform(test_inputs)
        x_test = []
        
        for i in range(lookback, len(test_inputs)):
            x_test.append(test_inputs[i-lookback:i, 0])
        
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        return x_train, y_train, x_test, y_test, scaler



    
    def RNN(self, dataset_name, units1, units2, units3, epochs, model):

        global x_train
        global x_test
        global y_train
        global y_test
            
        if model == "LSTM":
            regressor = Sequential()
            regressor.add(LSTM(units=units1, return_sequences=True, 
                               input_shape=(x_train.shape[1],1)))
            regressor.add(Dropout(0.2))
            
            if units3 > 0:
                regressor.add(LSTM(units=units2, return_sequences=True, 
                                  input_shape=(x_train.shape[1],1)))
                regressor.add(Dropout(0.2))
            
                regressor.add(LSTM(units=units3))
                regressor.add(Dropout(0.2))
            
            else:
                regressor.add(LSTM(units=units2))
                regressor.add(Dropout(0.2))
            
            regressor.add(Dense(units=1))
            # Compiling the RNN
            regressor.compile(optimizer='adam',loss='mean_squared_error', 
                              metrics=['mae','mse'])
            regressor.summary()
            
        
        else:
            regressor = Sequential()
            regressor.add(GRU(units=units1, return_sequences=True, 
                              input_shape=(x_train.shape[1],1)))
            regressor.add(Dropout(0.2))
            
            if units3 > 0:
                regressor.add(GRU(units=units2, return_sequences=True, 
                                  input_shape=(x_train.shape[1],1)))
                regressor.add(Dropout(0.2))
            
                regressor.add(GRU(units=units3))
                regressor.add(Dropout(0.2))
            
            else:
                regressor.add(GRU(units=units2))
                regressor.add(Dropout(0.2))
            
            regressor.add(Dense(units=1))
            # Compiling the RNN
            regressor.compile(optimizer='adam',loss='mean_absolute_error', 
                              metrics=['mae','mse'])
            regressor.summary()
            
        start = time.time()
        history = regressor.fit(x_train, y_train, epochs=epochs, batch_size=64, 
                                     verbose=1, 
                                     validation_split=0.2)
        print("Time taken to train the MLP %.1f seconds."%(time.time()-start))
    
        
        history = history.history
        epochs_plot = range(1, epochs + 1)
        
        plt.plot(epochs_plot, history['loss'], 'r', label='Training loss')
        plt.plot(epochs_plot, history['val_loss'], 'b', label='Validation loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.ylim(ymin=0)
        plt.show()
        
        plt.plot(epochs_plot, history['mse'], 'r', label='Training MSE')
        plt.plot(epochs_plot, history['val_mse'], 'b', label='Validation MSE')
        plt.title('Model MSE')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend()
        plt.ylim(ymin=0)
        plt.show()
        
        plt.plot(epochs_plot, history['mae'], 'r', label='Training MAE')
        plt.plot(epochs_plot, history['val_mae'], 'b', label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend()
        plt.ylim(ymin=0)
        plt.show()
        
        return(history, regressor)
    
    def RNN_test(self, dataset_name, model):
        global y_test
        
        RNN_pred = regressor.predict(x_test)
        RNN_pred = scaler.inverse_transform(RNN_pred)
        
        plt.figure(figsize = (16,6))
        plt.plot(y_test, color = 'blue', label = 'Real ' + dataset_name)
        plt.plot(RNN_pred, color = 'red', label = model + ' Forex Price Prediction For' + dataset_name)
        plt.title(dataset_name + ' Actual vs ' +model+' Prediction for '+ dataset_name, fontsize = 20,fontweight = "bold")
        plt.xlabel('Date', fontsize = 18,fontweight = "bold")
        plt.ylabel(dataset_name + ' Close Price ($)', fontsize = 18,fontweight = "bold")
        plt.legend()
        plt.grid()
        plt.show
        print("")
        print ('R Squared =',r2_score(y_test, RNN_pred))
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, RNN_pred)) 
        
        def mean_absolute_percentage_error(y_test, RNN_pred): 
            y_test, RNN_pred = np.array(y_test), np.array(RNN_pred)
            return np.mean(np.abs((y_test - RNN_pred) / y_test)) * 100  
        print('Mean Absolute Percentage Error:', mean_absolute_percentage_error(y_test, RNN_pred))
        
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, RNN_pred))
        
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, RNN_pred)))
        
        vals = RNN_pred.flatten().tolist()
        vals = {'Close': vals}
        new_df = pd.DataFrame(vals)
        
        y_vals = y_test.flatten().tolist()
        y_vals = {'y_Close': y_vals}
        y_new_df = pd.DataFrame(y_vals)
        y_new_df['y_pct_change'] = y_new_df['y_Close'].pct_change()
        
        combined = pd.concat([new_df, y_new_df], axis=1)
        combined = combined.assign(pred_actual_change=None)
        
        for i in range(1,len(combined)):
            x = ((combined['Close'][i]-combined['y_Close'][i-1])/combined['y_Close'][i-1])*100
            combined.at[i, 'pred_actual_change'] = x
            
        counter = 0
        for i in range(0,len(combined)):
            if pd.isna(combined['y_pct_change'][i]):
                continue
            
            if (combined['pred_actual_change'][i] > 0) & (combined['y_pct_change'][i] > 0):
                counter += 1
            
            elif (combined['pred_actual_change'][i] < 0) & (combined['y_pct_change'][i] < 0):
                counter += 1
        trend_counter = counter/len(combined)
        print("Same trend = " + str(trend_counter))
                
        return(history, RNN_pred, trend_counter)

#%%
