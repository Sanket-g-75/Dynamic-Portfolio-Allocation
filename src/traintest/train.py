import yaml
import os
import pandas as pd
import numpy as np
import mlflow
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow.pytorch

from dotenv import load_dotenv
load_dotenv()

path = os.environ['PROJECT_FOLDER']
os.chdir(path=path)

from src.models.lstm import lstm_model
from src.traintest.ddl import MyDataset, dataloader
from src.utils.lossfunction import SharpeLoss

with open(os.path.join(path,'config.yaml'),'r') as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Getting the data. As of now selecting stocks manually but later will build a program for it
dataset = pd.read_csv(os.path.join(path,'data','final','Dataset.csv'))
stocks = pd.read_csv(os.path.join(path,'data','final','Stocks.csv'))

dataset.set_index('Date',inplace=True)
stocks.set_index('Date',inplace=True)

stocks = stocks[['TATASTEEL.NS_7DReturn','SUNPHARMA.NS_7DReturn','RELIANCE.NS_7DReturn','INFY.NS_7DReturn','TATACONSUM.NS_7DReturn']]
stocks['Cash'] = -0.01

train_len = int(np.floor(0.7*len(dataset)))
val_len = len(dataset) - train_len

last_train_obs_date = dataset.iloc[[train_len-1],:].index[0]
last_val_obs_date = dataset.iloc[[-1],:].index[0]

# Dataset and Dataloader
train_dataset = MyDataset(
    X=dataset.iloc[:train_len,:],
    y=stocks.iloc[:train_len,:],
    lookback=config['metadata']['train_params']['lookback_period']
)
val_dataset = MyDataset(
    X=dataset.iloc[train_len+1:,:],
    y=stocks.iloc[train_len+1:,:],
    lookback=config['metadata']['train_params']['lookback_period']
)

train_loader = dataloader(
    dataset=train_dataset,
    batch_size=config['metadata']['train_params']['batch_size']
)
val_loader = dataloader(
    dataset=val_dataset,
    batch_size=config['metadata']['train_params']['batch_size']
)



# Model
model = lstm_model().to(device=device)

# Optimizier
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


# Epochs
num_epochs=config['metadata']['train_params']['epochs']

mlflow.set_experiment('Dynamic Portfolio Allocation')

mlflow.enable_system_metrics_logging()       # Log system metrics
mlflow.set_system_metrics_sampling_interval(1)   # Used this because the runs are very small it takes a very little time

with mlflow.start_run(run_name=str(datetime.now().strftime("%d/%m/%y-%H:%M:%S"))):
    
    metrics = []

    mlflow.log_param('Epochs',num_epochs)
    mlflow.log_param('Training Size',train_len)
    mlflow.log_param('Model Name',type(model).__name__)
    mlflow.log_param('Optimizer',type(optimizer))
    mlflow.log_param('Last Train Observation Date',last_train_obs_date)
    mlflow.log_param('Last Validation Observation Date',last_val_obs_date)

    for epoch in range(num_epochs):

        model.train()

        running_loss = 0

        for X, y in train_loader:
            
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            prediction = model(X)

            loss = SharpeLoss(prediction,y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()


        model.eval()

        val_loss = 0

        with torch.no_grad():

            for X1, y1 in val_loader:

                pred = model(X1)
                loss = SharpeLoss(pred, y1)
                val_loss += loss.item()

            val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss = {running_loss:.4f}, "
            f"Validation Loss = {val_loss:.4f}"
        )

        metrics.append([epoch+1,running_loss,val_loss])
    
    metrics = pd.DataFrame({'Epoch':metrics[0],'Train_Loss':metrics[1],'Val_Loss':metrics[2]})
    metrics.to_csv(os.path.join(path,'Metrics.csv'))
    
    torch.save(model.state_dict(), "model.pth")
    mlflow.log_artifact(local_path=os.path.join(path,'model.pth'),artifact_path='model')
    mlflow.log_artifact(local_path=os.path.join(path,'Metrics.csv'),artifact_path='metrics')