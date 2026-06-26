import os
import yaml
import numpy as np

from dotenv import load_dotenv
load_dotenv()

path = os.environ['PROJECT_FOLDER']

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyDataset(Dataset):
    def __init__(self,X,y,lookback=7):
        self.X = X
        self.y = y
        self.lookback = lookback

        self.X = self.X.reset_index(drop=True)
        self.y = self.y.reset_index(drop=True)
        self.X = torch.from_numpy(X.to_numpy(dtype=np.float32)).to(device)
        self.y = torch.from_numpy(y.to_numpy(dtype=np.float32)).to(device)
    
    def __len__(self):
        return (self.X.shape[0] - self.lookback + 1)
    
    def __getitem__(self,idx):
        X = self.X[idx: idx + self.lookback]
        y = self.y[idx: idx + self.lookback]

        return X,y

def dataloader(dataset,batch_size):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return dataloader