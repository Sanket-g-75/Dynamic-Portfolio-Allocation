import pandas as pd
import numpy as np
import os
import yfinance as yf
import datetime
import data.metadata as d


path = os.getcwd()



def download_new_ticker(ticker):
    data = yf.Tickers(ticker).history(period='10y')
    data = data.iloc[2:,:]
    data.rename(columns={'Price':'Date'},inplace=True)
    for i in data.columns:
        data.rename(columns={f'{i}':f'{ticker}-{i}'},inplace=True)
    data.to_csv(os.path.join(path,'data','raw',f'{ticker}.csv'))
    d.features.append(ticker)
   #  d.lastdate = data.iloc[[-1]]['Date']

def update_ticker(ticker,path):
    # data = yf.Tickers(ticker).history(period='10y')
    label = f'{ticker}.csv'
    data = pd.read_csv(os.path.join(path,'data','raw',label))
    data = data.iloc[2:,:]
    data.rename(columns={'Price':'Date'},inplace=True)
    for i in data.columns:
        if i != 'Date':
            data.rename(columns={f'{i}':f'{ticker}-{i}'},inplace=True)
    data.to_csv(os.path.join(path,'data','processed',f'p-{ticker}.csv'))
    # d.features.append(ticker)
   #  d.lastdate = data.iloc[[-1]]['Date']




def update_data():
    pass
