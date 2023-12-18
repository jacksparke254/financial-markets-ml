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

def calculate_rsi(prices, period=14):

    
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    
    # Calculate the Relative Strength (RS) by dividing the average gain by the average loss
    rs = avg_gain / avg_loss
    
    # Calculate the Relative Strength Index (RSI)
    rsi = 100 - (100 / (1 + rs))
    
    # Add the RSI to the dataframe
    return rsi



def calculate_macd(data, short_window=12, long_window=26, signal_window=9):

    # Calculate the short-term and long-term exponential moving averages
    short_ema = data.ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = data.ewm(span=long_window, min_periods=1, adjust=False).mean()

    # Calculate MACD line
    macd_line = short_ema - long_ema

    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_window, min_periods=1, adjust=False).mean()

    return macd_line, signal_line


def calculate_bollinger_bands(data, window=20, num_std=2):

    # Calculate the rolling mean (SMA)
    middle_band = data.rolling(window=window).mean()

    # Calculate the rolling standard deviation
    rolling_std = data.rolling(window=window).std()

    # Calculate upper and lower Bollinger Bands
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)

    return upper_band, middle_band, lower_band


def calculate_sma(data, window=20):

    sma = data.rolling(window=window).mean()
    return sma


def calculate_ema(data, window=20):

    ema = data.ewm(span=window, adjust=False).mean()
    return ema


def kaufman_efficiency_ratio(price_data):
    direction = price_data.diff(8).abs()
    volatility= price_data.diff().abs().rolling(8).sum()
    
    ker = direction/volatility
    
    return ker


def calculate_ema(data, window):
    # Calculate the smoothing factor (alpha)
    alpha = 2 / (window + 1)
    
    # Calculate the initial EMA using the simple moving average (SMA)
    ema_values = [data[0]]
    for i in range(1, len(data)):
        ema = alpha * data[i] + (1 - alpha) * ema_values[-1]
        ema_values.append(ema)
    
    return ema_values

def calculate_changes(lst):
    for i in range(1, len(lst)):
        change = lst[i] - lst[i - 1]
    return change


def madrid_ribbons(i, madrid):
    rib_list = []
    for key in list(madrid)[:-1]:
        change = calculate_changes(madrid[key][i-1:i+1])
        
        if change >= 0 and madrid[key][i] > madrid['100'][i]:
            rib_list.append(1)
        
        elif change <= 0 and madrid[key][i] < madrid['100'][i]:
            rib_list.append(0)
    
    if len(rib_list) == 18:
        if len(np.unique(rib_list)) == 1:
            if np.unique(rib_list)[0] == 1:
                return 2
            else:
                return 0
    else:
        return 1
    
def map_to_direction(value):
    if value < 0:
        return 'down'
    else:
        return 'up'
        
    
def generate_time_intervals(interval_hours):
    intervals = []
    start_time = pd.Timestamp('00:00:00')
    
    while start_time < pd.Timestamp('23:59:59'):
        intervals.append(start_time.strftime('%H:%M:%S'))
        start_time = start_time + pd.Timedelta(hours=interval_hours)
    
    return intervals


def adjust_class_labels(class_labels, num_missing_classes):
    missing_classes = [cls for cls in range(num_missing_classes) if cls not in class_labels]

    adjusted_labels = class_labels.copy()
    for i in range(len(adjusted_labels)):
        for missing_cls in missing_classes:
            if adjusted_labels[i] > missing_cls:
                adjusted_labels[i] -= 1

    # After the second missing class, decrement by 2
    second_missing_cls = max(missing_classes)
    for i in range(len(adjusted_labels)):
        if adjusted_labels[i] > second_missing_cls:
            adjusted_labels[i] -= 2

    return adjusted_labels

def find_missing_values(seq):
    """
    Finds the missing values in a sequence of sequential data.
    
    Parameters:
    seq (list): A list of sequential data.
    
    Returns:
    missing_values (list): A list of missing values in the sequence.
    """
    missing_values = []
    for i in range(seq[0], seq[-1] + 1):
        if i not in seq:
            missing_values.append(i)
    return missing_values




def reduce_sequence(missing_values, df):
    df_func = df.sort_values(by=2)
    df_func = df_func.reset_index(drop=True)
    index_1 = df_func[df_func[2] == missing_values[0]+1].index[0]
    df_1 = df_func[:index_1]

    for i in range(0,len(missing_values)):
        if i == 0:
                    
            index_missed = df_func[df_func[2] == missing_values[i]+1].index[0]
            df_spliced = df_func[index_missed:]
            df_spliced[2] = df_spliced[2] - 1 
            df_spliced = df_spliced.reset_index(drop=True)
            index_missed = df_spliced[df_spliced[2] == missing_values[i+1]+1].index[0]
            df_spliced = df_spliced[:index_missed]

        else:
                    
            index_missed = df_func[df_func[2] == missing_values[i]].index[0]
            #df_spliced = df_spliced[:index_missed]
            df_spliced2 = df_func[index_missed:]
            df_spliced2[2] = df_spliced2[2] - 1 
    
    df_fixed = pd.concat([df_1,df_spliced, df_spliced2], axis=0)
    df_fixed = df_fixed.reset_index(drop=True)
    df_fixed[2] = df_fixed[2] - 1
    
    return df_fixed

def reduce_sequence2(missing_values, combined_df):
    df_func = combined_df.sort_values(by=2)
    df_func = df_func.reset_index(drop=True)
    index_1 = df_func[df_func[2] == missing_values[0]+1].index[0]
    df_1 = df_func[:index_1]
    
    
    index_missed = df_func[df_func[2] == missing_values[0]+1].index[0]
    df_spliced = df_func[index_missed:]
    df_spliced[2] = df_spliced[2] - 1 
    df_spliced = df_spliced.reset_index(drop=True)
    index_missed = df_spliced[df_spliced[2] == missing_values[1]].index[0]
    df_spliced = df_spliced[:index_missed]
    
    
    index_missed = df_func[df_func[2] == missing_values[1]].index[0]
    #df_spliced = df_spliced[:index_missed]
    df_spliced2 = df_func[index_missed:]
    df_spliced2[2] = df_spliced2[2] - 1 
    
    df_fixed = pd.concat([df_1,df_spliced, df_spliced2], axis=0)
    df_fixed = df_fixed.reset_index(drop=True)
    df_fixed[2] = df_fixed[2] - 1
    return df_fixed



def pred_stats(RNN_pred, y_series, forex_data_minute, mae = True):
    vals = RNN_pred.flatten().tolist()
    vals = {'Preds': vals}
    new_df = pd.DataFrame(vals)
    
    
  
    y_vals = y_series.flatten().tolist()
    y_vals = {'y_Close': y_vals}
    y_new_df = pd.DataFrame(y_vals)
    y_new_df['y_pct_change'] = y_new_df['y_Close'].pct_change()
    
    combined = pd.concat([new_df, y_new_df], axis=1)
    combined = combined.assign(pred_actual_change=None)
    
    for i in range(1,len(combined)):
        x = ((combined['Preds'][i]-combined['y_Close'][i-1])/combined['y_Close'][i-1])*100
        combined.at[i, 'pred_actual_change'] = x
    
    trend_dict = {'var': []}
    
    counter = 0
    for i in range(0,len(combined)):
        if pd.isna(combined['y_pct_change'][i]):
            continue
        
        
        
        if (combined['pred_actual_change'][i] > 0) & (combined['y_pct_change'][i] > 0):
            counter += 1
            trend_dict['var'].append(1)
        
        elif (combined['pred_actual_change'][i] < 0) & (combined['y_pct_change'][i] < 0):
            counter += 1
            trend_dict['var'].append(1)
        
        else:
            trend_dict['var'].append(0)
            
            
            
            
    trend_counter = counter/len(combined)
    print("Same trend = " + str(trend_counter))
        
    print('\nAverage pct change ' + str(combined['pred_actual_change'].mean()))
    
    
    
    
    if mae == True:
        forex_time = forex_data_minute[30:]
        forex_time = forex_time.reset_index(drop=True)
        
        fx_pred_times = pd.concat([forex_time, new_df], axis=1)
        fx_pred_times['Close'] = fx_pred_times['Close'].astype(float)
    
        fx_pred_times['Error'] = fx_pred_times['Close'] - fx_pred_times['Preds']
        fx_pred_times['Date'] = pd.to_datetime(fx_pred_times['Date'])
        fx_pred_times['Preds'] = fx_pred_times['Preds'].round(3)
        # Set 'date' column as the index
        #fx_pred_times.set_index('Date', inplace=True)
        
        # Group by the hour and calculate the mean for each hour
        hourly_avg_error = fx_pred_times.groupby(fx_pred_times['Date'].dt.hour)['Error'].mean()
        
        # Display the result
        print(hourly_avg_error)
        return 
    
    


    forex_trend_df = forex_data_minute[31:]
    forex_trend_df = forex_trend_df.reset_index(drop=True)  
    
    trends = pd.DataFrame(trend_dict)
    added_df = pd.concat([forex_trend_df, trends], axis=1)
    added_df['Date'] = pd.to_datetime(added_df['Date'])
    
    
    added_df['3_hour_period'] = added_df['Date'].dt.floor('3H')
    
    period_counts = added_df.groupby('3_hour_period')['var'].sum()

    # Step 3: Find the 3-hour period with the most '1's
    max_ones_period = period_counts.idxmax()
    max_ones_count = period_counts.max()
    
    print("3-hour period with the most '1's:")
    print("Period:", max_ones_period)
    print("Count of '1's:", max_ones_count)
    
    int_dict = {1:'1H',
                2: '2H',
                3: '3H',
                4: '4H',
                6: '6H'
 
        }
    

    
    for i in [1,2,3,4,6]:
        
        u = int_dict[i]
        
        added_df = pd.concat([forex_trend_df, trends], axis=1)
        added_df['Date'] = pd.to_datetime(added_df['Date'])
        added_df['3_hour_period'] = added_df['Date'].dt.floor(u)
    
        period_counts = added_df.groupby('3_hour_period')['var'].sum()
        
        
        # Set the desired interval in hours
        interval_hours = i
        
        # Generate starting times for time intervals with the specified interval
        time_intervals = generate_time_intervals(interval_hours)
    

    
        averages_at_times = {time: [] for time in time_intervals}
        
        # Calculate the average count of '1's for the specified times within each 3-hour period
        for index, row in added_df.iterrows():
            for time in time_intervals:
                if row['Date'].strftime('%H:%M:%S') == time:
                    averages_at_times[time].append(period_counts[row['3_hour_period']])
        
        # Calculate the average for each specified time
        for time, counts in averages_at_times.items():
            averages_at_times[time] = np.mean(counts)
        
        # Print the averages for the specified times
        print("Averages at specified times:", i)
        for time, average_count in averages_at_times.items():
            print(f"{time}: {average_count}")
        print('\n')
        
        return