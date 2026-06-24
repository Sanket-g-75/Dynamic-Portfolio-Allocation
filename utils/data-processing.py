import os
import pandas as pd
import numpy as np
import dotenv
import yaml

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
    data.set_index('Date',inplace=True)

    if i in features:   # Normalization
        data = pd.DataFrame(
            scaler.fit_transform(data),
        columns=data.columns,
        index=data.index
        )

    data.to_csv(os.path.join(path,'data','processed',f'p-{i}.csv'))