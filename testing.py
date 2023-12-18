# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:06:51 2023

@author: jacks
"""

forex_list = ['EUR_USD', 'USD_CNH', 'USD_JPY', 'XAU_USD',"EUR_JPY", "GBP_USD", 'GBP_JPY']



api_key = 
accountID = 


client = API(access_token=api_key)

params = {
  "count": 5000,
  "granularity": 'M15'
}


timestamp = pd.Timestamp('2023-10-13 21:00:00+0000', tz='UTC')
for sym in forex_list:

    dataset_name = sym + "_15"
    r = instruments.InstrumentsCandles(instrument=sym, params=params)
    client.request(r)
    
    
    for i in range(30,len(r.response['candles'])):
        if i == 30:
            series = pd.DataFrame(columns=['Date', "Close"], index=range(30,len(r.response['candles'])))
            series['Date'][i] = pd.to_datetime(r.response['candles'][0]['time'])
            series['Close'][i] = float(r.response['candles'][i]['mid']['c'])
        else:
            series['Date'][i] = pd.to_datetime(r.response['candles'][i]['time'])
            series['Close'][i] = float(r.response['candles'][i]['mid']['c'])
    

    
    forex_data_minute = series

    
    forex_data = forex_data_minute.filter(["Close"])
    
    scaler = MinMaxScaler(feature_range=(0,1))
        
    forex_data = scaler.fit_transform(forex_data.values)
    
    
    series = []
    y_series = []
    for i in range(30, len(forex_data)):
        series.append(forex_data[i-30:i, 0])
        y_series.append(forex_data[i, 0])
    
    x_series, y_series = np.array(series), np.array(y_series)
    x_series = np.reshape(x_series, (x_series.shape[0], x_series.shape[1], 1))
    
    
    RNN_pred = regressor.predict(x_series)
    RNN_pred = scaler.inverse_transform(RNN_pred)
    #RNN_pred = RNN_pred*0.99995

    
    
    
    y_series = scaler.inverse_transform(y_series.reshape(len(y_series),1))
    
    
    plt.figure(figsize = (16,6))
    plt.plot(y_series, color = 'blue', label = 'Real ' + dataset_name)
    plt.plot(RNN_pred, color = 'red', label = 'Forex Price Prediction For' + dataset_name)
    plt.title(dataset_name + ' Actual vs model Prediction for '+ dataset_name, fontsize = 20,fontweight = "bold")
    plt.xlabel('Date', fontsize = 18,fontweight = "bold")
    plt.ylabel(dataset_name + ' Close Price ($)', fontsize = 18,fontweight = "bold")
    plt.legend()
    plt.grid()
    plt.show
    print("")
    print('symbol:', sym)
    print ('R Squared =',r2_score(y_series, RNN_pred))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_series, RNN_pred))
    
    mae_perc = metrics.mean_absolute_error(y_series, RNN_pred)/float(forex_data_minute['Close'].tail(1))

    print('mae perc:',mae_perc)
    
    pred_stats(RNN_pred, y_series, forex_data_minute, mae = True)
    
    
