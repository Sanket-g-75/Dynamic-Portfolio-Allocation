import os
import pandas as pd
import numpy as np
import dotenv
import yaml
import math

from sklearn.preprocessing import StandardScaler


dotenv.load_dotenv()
path = os.environ['PROJECT_FOLDER']


with open(os.path.join(path,'config.yaml'),'r') as f:
    config = yaml.safe_load(f)



stocks = []
for i in range(len(config['stocks'])):
    stocks.append(config['stocks'][i][0])

features = config['features']['commodities'] + config['features']['forex'] + config['features']['w_indices'] + config['features']['other']
instruments = stocks + features

scaler = StandardScaler()


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
        data[f'{i}_7DReturn'] = data[f'{i}_7DReturn'].fillna(0)

    data.drop(columns=[f'{i}_Open',f'{i}_High',f'{i}_Low',f'{i}_Close'],inplace=True)

    data.to_csv(os.path.join(path,'data','processed',f'p-{i}.csv'))