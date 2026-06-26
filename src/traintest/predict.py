import pandas as pd
import numpy as np
import os
import yaml
from dotenv import load_dotenv

import torch
import torch.nn as nn

from src.models.lstm import lstm_model
from src.utils.prep_train_data import create_pred_data


load_dotenv()

path = os.environ['PROJECT_FOLDER']
os.chdir(path)

with open(os.path.join(path,'config.yaml'),'r') as f:
    config = yaml.safe_load(f)

# Making a list of all the stocks and features
stocks = []
features = []

for i in range(len(config['stocks'])):
    stocks.append(config['stocks'][i][0])

features = config['features']['w_indices'] + config['features']['commodities'] + config['features']['forex'] + config['features']['other']


# Loading the model
model = lstm_model()
model.load_state_dict(torch.load(os.path.join(path,'model.pth')))
model.eval()


date,X,y = create_pred_data(config=config,features = features, stocks=stocks)

print([date,model(X)])