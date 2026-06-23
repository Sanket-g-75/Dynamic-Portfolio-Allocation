import yfinance as yf
import pandas as pd
import numpy as np
import os
import yaml
from datetime import date

os.chdir(r'D:\1.Work\WorkStation\WorkSpace\Projects\Dynamic-Portfolio-Allocation')
path = os.getcwd()


# Loading the configs
with open(os.path.join(path,'config.yaml'),'r') as f:
    config = yaml.safe_load(f)


# Making a list of all the stocks and features
stocks = []
features = []

for i in range(len(config['stocks'])):
    stocks.append(config['stocks'][i][0])

features = config['features']['w_indices'] + config['features']['commodities'] + config['features']['forex'] + config['features']['other']


# Downloading the data
for i in (stocks + features):
    data = yf.Ticker(i).history(period='10y').rename(
        columns={"Open": f'{i}_Open',
                'High':f'{i}_High',
                "Low":f'{i}_Low',
                "Close":f'{i}_Close',
                "Volume":f'{i}_Volume',
                "Dividends":f'{i}_Dividends',
                "Stock Splits":f'{i}_StockSplits'}
        )
    data.to_csv(os.path.join(path,'data','raw',f'{i}.csv'))


# Update the date in the config file
config['metadata']['data_present_till'] = str(pd.to_datetime(date.today()))
 
with open(os.path.join(path,'config.yaml'),'w') as f:
    config = yaml.safe_dump(config,f)