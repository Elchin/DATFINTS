# This script produces nice visual of a specified stock (by default BTC-USD)
# To run the script use cpy and paste the line below
# $ streamlit run "crypto_dashboard.py"
# to fix streamlight failure cd ~/.streamlit , and delete: rm config.toml

import streamlit as st
import datetime
import pandas as pd
import plotly.graph_objects as go
from pandas_datareader import data as wb
from PIL import Image 

st.write("""
#Stock Market/Crypto dashboard application
""")

st.sidebar.header("User Input")

def get_input():
    start_date=st.sidebar.text_input("Start-date","2020-01-01")
    end_date=st.sidebar.text_input("End-date","2020-06-01")
    crypto_symbol=st.sidebar.text_input("Stock Symbol","BTC-USD")
    return start_date, end_date, crypto_symbol

def get_crypto_name(symbol):
    symbol=symbol.upper()
    #if symbol=="BTC-USD":
    #    return "Bitcoin"
    #else:
    #    return None
    return symbol
    
def get_date(symbol, start_date, end_date):
    symbol=symbol.upper()
    stock=wb.DataReader(symbol,start=start_date,data_source='yahoo')
        
    return stock.loc[start_date:end_date]

start,end,symbol = get_input()
stock=get_date(symbol, start, end)
crypto_name=get_crypto_name(symbol)

fig = go.Figure(
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

st.header(crypto_name+"Data")
st.write(stock)

st.header(crypto_name+"Close Price")
st.line_chart(stock['Close'])

st.header(crypto_name+"Volume")
st.bar_chart(stock['Volume'])

st.header(crypto_name+"Candle Stick")
st.plotly_chart(fig)

