import yaml
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from dotenv import load_dotenv
load_dotenv()

from src.models.lstmv1 import lstm_model
from src.traintest.ddl import MyDataset, dataloader
from src.utils.lossfunction import loss_function

path = os.environ['PROJECT_FOLDER']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Getting the data. As of now selecting stocks manually but later will build a program for it
dataset = pd.read_csv(os.path.join(path,'data','final','Dataset.csv'))
stocks = pd.read_csv(os.path.join(path,'data','final','Stocks.csv'))

dataset.set_index('Date',inplace=True)
stocks.set_index('Date',inplace=True)

stocks = stocks[['TATASTEEL.NS_7DReturn','SUNPHARMA.NS_7DReturn','RELIANCE.NS_7DReturn','INFY.NS_7DReturn','TATACONSUM.NS_7DReturn']]

# Dataset and Dataloader
dataset = MyDataset(X=dataset,y=stocks)
dataloader = dataloader(dataset=dataset,batch_size=24)
model = lstm_model().to(device=device)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

num_epochs=35


for epoch in range(num_epochs):

    model.train()

    running_loss = 0

    for X, y in dataloader:
        
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        prediction = model(X)

        loss = loss_function(prediction,y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}: "
        f"{running_loss/len(dataloader):.4f}"
    )