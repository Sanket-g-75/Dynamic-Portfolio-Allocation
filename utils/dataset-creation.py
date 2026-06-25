import os
import pandas as pd
import numpy as np
import math
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()

path = os.environ['PROJECT_FOLDER']

with open(os.path.join(path,'config.yaml')) as f:
    config = yaml.safe_load(f)

stocks = []
for i in range(len(config['stocks'])):
    stocks.append(config['stocks'][i][0])

features = config['features']['commodities'] + config['features']['forex'] + config['features']['w_indices'] + config['features']['other']
instruments = stocks + features
instruments.remove('JIOFIN.NS')
instruments.remove('TMCV.NS')

common_idx = None

for i in instruments:
    data = pd.read_csv(os.path.join(path,'data','processed',f'p-{i}.csv'))
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date',inplace=True)
    common_idx = data.index if common_idx is None else common_idx.intersection(data.index)

common_idx = common_idx.sort_values()

dataframes = []
fstocks = []

for i in features:
    data = pd.read_csv(os.path.join(path,'data','processed',f'p-{i}.csv'))
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date',inplace=True)
    data.drop(columns=f'{i}_Volume',inplace=True)              # As of now will remove the volumes
    data = data.loc[common_idx]
    dataframes.append(data)

for i in ["TATASTEEL.NS", "SUNPHARMA.NS", "RELIANCE.NS", "INFY.NS", "TATACONSUM.NS"]:
    data = pd.read_csv(os.path.join(path,'data','processed',f'p-{i}.csv'))
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date',inplace=True)
    data.drop(columns=f'{i}_Volume',inplace=True) 
    data = data.loc[common_idx]
    fstocks.append(data)

dataset = pd.concat(dataframes,axis=1)
stks = pd.concat(fstocks,axis=1)

dataset.to_csv(os.path.join(path,'data','final','Dataset.csv'))
stks.to_csv(os.path.join(path,'data','final','Stocks.csv'))