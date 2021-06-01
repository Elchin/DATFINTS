import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import datfints_utils as df_utils
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def calc_rsi(stock_index,date='2021-1-1'):
    '''
    Calculates stock relative strength metric (RSI)
    that helps to identify if stock is overbought or oversold
    Higher RSI indicate stock is overbought, lower RSI means the opposite. 
    Inputs:
        stock_index: e.g. 'ETH-USD'
        date:        e.g. '2021-1-1'
    '''

    stock=wb.DataReader(stock_index,start=date,data_source='yahoo') #stock ticker index
    delta=stock['Adj Close'].diff(1)
    delta=delta.dropna()

    up = delta.copy()
    down = delta.copy()

    up[up<0]=0
    down[down>0]=0

    period=14 # number of days
    #Average gains and lossess
    ave_gain = up.rolling(window=period).mean()
    ave_loss = abs(down.rolling(window=period).mean())

    RS=ave_gain/ave_loss
    RSI=100.0-(100.0 / (1.0+RS))

    plt.figure(figsize=(12.2,4.5))
    plt.plot(stock.index,stock['Adj Close'])
    plt.title('Adj. Close '+stock_index)
    plt.ylabel('Price [$]')

    plt.figure(figsize=(12.2,4.5))
    RSI.plot()
    plt.title('Relative Strength Index  (RSI)')
    plt.ylabel('RSI [%]')
    plt.axhline(0,linestyle='--', alpha=0.5,color='gray')
    plt.axhline(10,linestyle='--', alpha=0.5,color='orange')
    plt.axhline(20,linestyle='--', alpha=0.5,color='green')
    plt.axhline(30,linestyle='--', alpha=0.5,color='red')
    plt.axhline(70,linestyle='--', alpha=0.5,color='red')
    plt.axhline(80,linestyle='--', alpha=0.5,color='green')
    plt.axhline(90,linestyle='--', alpha=0.5,color='orange')
    plt.axhline(100,linestyle='--', alpha=0.5,color='gray')
    
    return

def stock_price_prediction(stock_index,date='2021-1-1'):
    '''
    Predicts stock price 14 days in advance 
    using LinearRegression model
    Inputs:
        stock_index: e.g. 'ETH-USD'
        date:        e.g. '2021-1-1'
    '''
    #stock=wb.DataReader('ETH-USD',start='2020-1-1',data_source='yahoo') 
    stock=wb.DataReader(stock_index,start=date,data_source='yahoo') #stock ticker index
    
    projection=14 # how many days into the future
    stock['predictions']=stock[['Close']].shift(-projection)
    X=np.array(stock[['Close']])
    X=X[:-projection]
    y=stock['predictions'].values
    y=y[:-projection]
    
    # split the data into 85% training and 15% testing
    x_train, x_test, y_train, y_test = train_test_split (X, y, test_size=0.15) 
    LinReg=LinearRegression()
    LinReg.fit(x_train,y_train)
    
    #test the model score
    linReg_confidence=LinReg.score(x_test,y_test)
    print('Linear Regression Confidence:',linReg_confidence)
    
    #predict lasrt 14 days of the stock
    x_projection = np.array(stock[['Close']])[-projection:]
    
    # Linear regression model preductions
    LinReg_prediction=LinReg.predict(x_projection)
    
    
    return LinReg_prediction