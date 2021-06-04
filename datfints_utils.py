import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import datfints_utils as df_utils
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# All functions described below can be found on this channel
# https://www.youtube.com/channel/UCbmb5IoBtHZTpYZCDBOC1CA

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

def on_balance_volume(stock_index,date='2021-1-1'):
    '''
    On Balance Volume shows when to buy or sell the stock 
    Inputs:
        stock_index: e.g. 'ETH-USD'
        date:        e.g. '2021-1-1'
    '''

    stock=wb.DataReader(stock_index,start=date,data_source='yahoo') #stock ticker index
    
    #Calculate on Balance Volume
    OBV=[]
    OBV.append(0)
    for i in range(1,len(stock.Close)):
        if stock.Close[i] > stock.Close[i-1]:
            OBV.append(OBV[-1]+stock.Volume[i])
        elif stock.Close[i] < stock.Close[i-1]:
            OBV.append(OBV[-1]-stock.Volume[i])
        else:
            OBV.append(OBV[-1])

    # EMA exponential moving average
    stock['OBV']=OBV
    stock['OBV_EMA']=stock['OBV'].ewm(span=20).mean()
    stock

    plt.figure(figsize=(12.2,4.5))
    plt.plot(stock.index,stock['OBV'],label='OBV')
    plt.plot(stock.index,stock['OBV_EMA'],label='OBV_EMA')

    x=buy_sell(stock,'OBV','OBV_EMA')
    stock['Buy_Signal_Price']=x[0]
    stock['Sell_Signal_Price']=x[1]

    plt.figure(figsize=(12.2,4.5))
    plt.plot(stock.index,stock['Close'],label='Close',alpha=0.35)
    plt.scatter(stock.index,stock['Buy_Signal_Price'],label='Buy',marker='^',alpha=1,color='green')
    plt.scatter(stock.index,stock['Sell_Signal_Price'],label='Sell',marker='v',alpha=1,color='red')
    plt.legend()
    plt.ylabel('Price [$]')
    plt.xlabel('Date [d]')

    return 

#if OBV > OBV_EMA -> buy
#if OBV < OBV_EMA -> buy
def buy_sell(data,col1,col2):
    '''
    col1: OBV 
    col2: OBV-EMA
    '''
    SigPriceBuy=[]
    SigPriceSell=[]
    flag=1
    for i in range(0,len(data)):
        if data[col1][i] > data[col2][i] and flag != 1:
            SigPriceBuy.append(data['Close'][i])
            SigPriceSell.append(np.nan)
            flag=1
        elif data[col1][i] > data[col2][i] and flag != 0:
            SigPriceSell.append(data['Close'][i])
            SigPriceBuy.append(np.nan)
            flag=0
        else:
            SigPriceBuy.append(np.nan)
            SigPriceSell.append(np.nan)
            
    return (SigPriceBuy,SigPriceSell)

def plot_candlesticks(stock_index,date='2021-1-1'):
    '''
    Plot Japanese Candle Sticks in the interactive mode 
    Make sure that you have plotly package installed
    e.g. 'pip install plotly'
    Inputs:
        stock_index: e.g. 'ETH-USD'
        date:        e.g. '2021-1-1'
    '''
    stock=wb.DataReader(stock_index,start=date,data_source='yahoo') #stock ticker index
    #https://www.youtube.com/watch?v=4fhBXFSS1lc
    figure = go.Figure(
        data = [
            go.Candlestick(
                x = stock.index,
                low = stock['Low'],
                high = stock['High'],
                close = stock['Close'],
                open = stock['Open'],
                increasing_line_color = 'green',
                decreasing_line_color = 'red'
            )
        ]
    )

    figure.update_layout(
        title=stock_index,
        yaxis_title='Price ($)',
        xaxis_title='Time'
    )

    figure.show()

def FRL(stock_index='BTC-USD',date='2021-1-1'):
    ''' 
    Calculates and plots the Fibonacci Retracement levels with no Fib level raios of 0.5 50%
    Fibonacci  ratios 0.236, .382, 0.618
    E.g. 1,1,2,3,5,8,13,21,34,55,89,144  ==> 89/144=0.618
    to get .382 take any number i and divide by i+2 i.e.  21/55~0.38
    Inputs:
        stock_index: e.g. 'ETH-USD'
        date:        e.g. '2021-1-1'
    '''
    
    stock=wb.DataReader(stock_index,start=date,data_source='yahoo') #stock ticker index

    max_price=stock['Close'].max()
    min_price=stock['Close'].min()
    diff=max_price-min_price
    level1=max_price-diff*0.236
    level2=max_price-diff*0.382
    level3=max_price-diff*0.5
    level4=max_price-diff*0.618

    print('level percentage  price ($)')
    print('00.0%\t\t',max_price)
    print('23.6%\t\t',level1)
    print('38.2%\t\t',level2)
    print('50.0%\t\t',level3)
    print('61.8%\t\t',level4)
    print('100.%\t\t',min_price)

    #new_stock=stock
    fig=plt.figure(figsize=(12.2,4.5))

    ax1=fig.add_subplot(1,1,1)
    plt.plot(stock.index,stock['Close'])
    plt.title('Fibonacci Retracement Levels '+stock_index)
    plt.ylabel('Price [$]')

    plt.axhline(max_price,linestyle='--', alpha=0.5,color='red')
    plt.axhline(level1,linestyle='--', alpha=0.5,color='orange')
    plt.axhline(level2,linestyle='--', alpha=0.5,color='yellow')
    plt.axhline(level3,linestyle='--', alpha=0.5,color='green')
    plt.axhline(level4,linestyle='--', alpha=0.5,color='blue')
    plt.axhline(min_price,linestyle='--', alpha=0.5,color='purple')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')

    #new_stock=stock
    fig=plt.figure(figsize=(12.2,4.5))
    ax2=fig.add_subplot(1,1,1)
    plt.plot(stock.index,stock['Close'],color='black')
    plt.title('Fibonacci Retracement Levels '+stock_index)
    plt.ylabel('Price [$]')

    plt.axhline(max_price,linestyle='--', alpha=0.5,color='red')
    ax2.fill_between(stock.index,max_price,level1,color='red')

    plt.axhline(level1,linestyle='--', alpha=0.5,color='orange')
    ax2.fill_between(stock.index,level1,level2,color='orange')

    plt.axhline(level2,linestyle='--', alpha=0.5,color='yellow')
    ax2.fill_between(stock.index,level2,level3,color='yellow')

    plt.axhline(level3,linestyle='--', alpha=0.5,color='green')
    ax2.fill_between(stock.index,level3,level4,color='green')

    plt.axhline(level4,linestyle='--', alpha=0.5,color='blue')
    ax2.fill_between(stock.index,level4,min_price,color='blue')

    plt.axhline(min_price,linestyle='--', alpha=0.5,color='purple')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    
    return