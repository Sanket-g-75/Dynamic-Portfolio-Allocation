import os
import pandas as pd
import numpy as np
import math
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from datetime import datetime
import torch
from typing import Literal
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

load_dotenv()
path = os.environ['PROJECT_FOLDER']

with open(os.path.join(path,'config.yaml')) as f:
    config = yaml.safe_load(f)

# Data Download
# Making a list of all the stocks and features
stocks = []
features = []

for i in range(len(config['stocks'])):
    stocks.append(config['stocks'][i][0])

features = config['features']['w_indices'] + config['features']['commodities'] + config['features']['forex'] + config['features']['other']
instruments = stocks + features
instruments.remove('JIOFIN.NS')
instruments.remove('TMCV.NS')

# Downloading the data
def data_download(
    path:str,
    ticker:str,
    config: dict,
    period:Literal['1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'] = 'max'
):
    data = yf.Ticker(i).history(period='max').rename(
        columns={"Open": f'{i}_Open',
                'High':f'{i}_High',
                "Low":f'{i}_Low',
                "Close":f'{i}_Close',
                "Volume":f'{i}_Volume',
                "Dividends":f'{i}_Dividends',
                "Stock Splits":f'{i}_StockSplits'}
        )
    data.to_csv(os.path.join(path,f'{ticker}_{period}.csv'))

    # Update the date in the config file
    config['metadata']['data_present_till'] = str(pd.to_datetime(date.today()))


# Dataset Creation
# Creating a common_id
def common_idx(
    sets:list[list[str]] = [stocks,features]
) -> list[datetime]:
    
    common_idx = None

    instruments = []
    for sublist in sets:
        instruments.append(sublist)
    instruments.remove('JIOFIN.NS')
    instruments.remove('TMCV.NS')

    for i in instruments:
        data = pd.read_csv(os.path.join(path,'data','processed',f'p-{i}.csv'))
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date',inplace=True)
        common_idx = data.index if common_idx is None else common_idx.intersection(data.index)

    common_idx = common_idx.sort_values()
    return common_idx

# Create Final Datasets
def create_datasets(
    features:list,
    common_idx: list[datetime],
    stocks:list = ["TATASTEEL.NS", "SUNPHARMA.NS", "RELIANCE.NS", "INFY.NS", "TATACONSUM.NS"],
) -> tuple(pd.DataFrame,pd.DataFrame):
    dataframes = []
    fstocks = []

    for i in features:
        data = pd.read_csv(os.path.join(path,'data','processed',f'p-{i}.csv'))
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date',inplace=True)
        data.drop(columns=f'{i}_Volume',inplace=True)              # As of now will remove the volumes
        data = data.loc[common_idx]
        dataframes.append(data)

    for i in stocks:
        data = pd.read_csv(os.path.join(path,'data','processed',f'p-{i}.csv'))
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date',inplace=True)
        data.drop(columns=f'{i}_Volume',inplace=True) 
        data = data.loc[common_idx]
        fstocks.append(data)

    dataset = pd.concat(dataframes,axis=1)
    stks = pd.concat(fstocks,axis=1)

    return dataset, stks    # This by default returns a tuple


# Feature Creation
def feature_creation(
    sets:list[list[str]],
    path: str
) -> tuple[list,torch.Tensor, torch.Tensor]:
    instruments = []

    for i in sets:
        instrument.append(i)

    for i in instruments:
        data = pd.read_csv(os.path.join(path,'data','raw',f'{i}.csv'))
        data.drop(columns=[f'{i}_Dividends',f'{i}_StockSplits'],inplace=True)

        data[f'{i}_HL'] = (data[f'{i}_High']-data[f'{i}_Low'])
        data[f'{i}_OC'] = (data[f'{i}_Close']-data[f'{i}_Open'])
        
        data[f'{i}_Volatility'] = np.sqrt(data[f'{i}_HL']**2 + data[f'{i}_OC']**2)
        data[f'{i}_Theta'] = np.arctan(data[f'{i}_OC']/data[f'{i}_HL'])
        data[f'{i}_Theta'] = data[f'{i}_Theta'].apply(lambda x: ((x * 180) / math.pi)) 
        data.fillna(0,inplace=True) 
        
        data['Date'] = pd.to_datetime(data['Date'],utc=True).dt.date
        data.set_index('Date',inplace=True)

        if i in features:   # Normalization
            data = pd.DataFrame(
                scaler.fit_transform(data),
            columns=data.columns,
            index=data.index
            )

        # Adding 7D return
        if i in stocks:
            data[f'{i}_7DReturn'] = (100*(data[f'{i}_Close'].shift(-7) - data[f'{i}_Close'])/data[f'{i}_Close'])
            # data[f'{i}_7DReturn'] = data[f'{i}_7DReturn'].fillna(0)   > Instead of doing this, will remove the last lookback window
            data.dropna(axis=0,inplace=True)

        data.drop(columns=[f'{i}_Open',f'{i}_High',f'{i}_Low',f'{i}_Close'],inplace=True)
        data.to_csv(os.path.join(path,f'p-{i}.csv'))



# Create the Prediction Data
def create_prediction_data(
    sets:list[list[str]] = [features, stocks],
    config:dict = config,
    period:str = '1mo'
) -> tuple[list, torch.Tensor, torch.Tensor]:
    instruments = []
    for i in sets:
        instrument.append(i)
    
    # Download the last 1 month data in a list
    dfs = []
    for i in instruments:
        
        # Write an exception to keep on running the loop until all the stocks data is not downloaded
        data = yf.Ticker(i).history(period=period).rename(
            columns={"Open": f'{i}_Open',
                    'High':f'{i}_High',
                    "Low":f'{i}_Low',
                    "Close":f'{i}_Close',
                    "Volume":f'{i}_Volume',
                    "Dividends":f'{i}_Dividends',
                    "Stock Splits":f'{i}_StockSplits'}
            )
        dfs.append(data)


    # Processing Data
    scaler = StandardScaler()    

    dfs2 = []
    for data in dfs:
        i = data.columns[2].split("_")[0]
        data.drop(columns=[f'{i}_Dividends',f'{i}_StockSplits'],inplace=True)

        data[f'{i}_HL'] = (data[f'{i}_High']-data[f'{i}_Low'])
        data[f'{i}_OC'] = (data[f'{i}_Close']-data[f'{i}_Open'])
        
        data[f'{i}_Volatility'] = np.sqrt(data[f'{i}_HL']**2 + data[f'{i}_OC']**2)
        data[f'{i}_Theta'] = np.arctan(data[f'{i}_OC']/data[f'{i}_HL'])
        data[f'{i}_Theta'] = data[f'{i}_Theta'].apply(lambda x: ((x * 180) / math.pi)) 
        data.fillna(0,inplace=True) 
        

        if i in features:   # Normalization
            data = pd.DataFrame(
                scaler.fit_transform(data),
            columns=data.columns,
            index=data.index
            )

        # Adding 7D return
        if i in stocks:
            data[f'{i}_7DReturn'] = (100*(data[f'{i}_Close'].shift(-7) - data[f'{i}_Close'])/data[f'{i}_Close'])
            # data[f'{i}_7DReturn'] = data[f'{i}_7DReturn'].fillna(0)   > Instead of doing this, will remove the last lookback window
            data.dropna(axis=0,inplace=True)

        data.drop(columns=[f'{i}_Open',f'{i}_High',f'{i}_Low',f'{i}_Close'],inplace=True)

        dfs2.append(data)

    # Final Dataset creation
    dfs3 = []
    common_idx = None

    for data in dfs2:
        i = data.columns[2].split("_")[0]
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'],utc=True).dt.date
        data.set_index('Date',inplace=True)
        common_idx = data.index if common_idx is None else common_idx.intersection(data.index)

    common_idx = common_idx.sort_values()

    dataframes = []
    fstocks = []

    for data in dfs2:
        i = data.columns[2].split("_")[0]
        data.drop(columns=f'{i}_Volume',inplace=True) 
        if i in features:
            data = data.loc[common_idx]
            dataframes.append(data)
        elif i in stocks:
            data = data.loc[common_idx]
            fstocks.append(data)
        else:
            pass

    dataset = pd.concat(dataframes,axis=1)
    stks = pd.concat(fstocks,axis=1)


    stks = stks[['TATASTEEL.NS_7DReturn','SUNPHARMA.NS_7DReturn','RELIANCE.NS_7DReturn','INFY.NS_7DReturn','TATACONSUM.NS_7DReturn']]
    stks['Cash'] = -0.01

    dataset = dataset.iloc[-7:,:]
    stks = stks.iloc[-7:,:]

    X = torch.tensor(dataset.values,dtype=torch.float32)
    y = torch.tensor(stks.values,dtype=torch.float32)

    return common_idx.to_list()[-1],X.unsqueeze(dim=1),y

