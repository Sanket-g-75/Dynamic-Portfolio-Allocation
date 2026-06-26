import yfinance as yf
import pandas as pd
import numpy as np
import os
import yaml
from datetime import date
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
import math
import torch


load_dotenv()
path = os.environ['PROJECT_FOLDER']

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
instruments = stocks+features
instruments.remove('JIOFIN.NS')
instruments.remove('TMCV.NS')

# Prepare pred data
def create_pred_data(config=config,features=features,stocks=stocks):  # Downloads the last 1 month data
    instruments = stocks+features
    instruments.remove('JIOFIN.NS')
    instruments.remove('TMCV.NS')
    
    dfs = []
    for i in instruments:

        # Download the last 1 month data.
        # Write an exception to keep on running the loop until all the stocks data is not downloaded
        data = yf.Ticker(i).history(period='1mo').rename(
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

# date,X,y = create_pred_data(config=config,features=features,stocks=stocks)
# print(X)
# print(y)