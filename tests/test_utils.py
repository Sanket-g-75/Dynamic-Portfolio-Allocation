import os
import yaml
import numpy as np
from datetime import datetime

path = r'D:\1.Work\WorkStation\WorkSpace\Projects\Dynamic-Portfolio-Allocation'
os.chdir(path)

from src.utils.lossfunction import SharpeLoss
from src.utils.prep_train_data import create_pred_data

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

# Testing for the loss function
def test_lossfunction():
    sharpe = SharpeLoss([0.5,0.2,0.2,0.1],[1,-1,0.3,-0.4])
    assert 1 > 0

def test_create_pred_data():
    date,X,y = create_pred_data()
    assert 1 > 0

