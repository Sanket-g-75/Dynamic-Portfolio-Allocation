import os
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler


os.chdir(r"D:\1.Work\WorkStation\WorkSpace\Projects\Dynamic-Portfolio-Allocation")
path = os.getcwd()

# combine tables from multiple with common index
# so I will have to create a new dataset in the processed section where I will create a combined features for all of them
# write a function for this

def combine_data(features): # Finds data from the processed folder and then combines them
    df = pd.DataFrame()

    for i in features:
        if df.empty:
            df = pd.read_csv(os.path.join(path,'data','processed',f'p-{i}.csv'),index_col=0)
            df.set_index('Date',inplace=True)
        else:
            right = pd.read_csv(os.path.join(path,'data','processed',f'p-{i}.csv'),index_col=0)
            right.set_index('Date',inplace=True)
    
            df = pd.concat([df,right],join='inner',axis=1)

    name = f'{datetime.date.today()}-Dataset.csv'
    df.to_csv(os.path.join(path,'data','final',name))

def scaling(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled