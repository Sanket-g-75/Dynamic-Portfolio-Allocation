
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class lstm_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=76,
            hidden_size=35,
            num_layers=1,    # This adds sequential layers where each layer takes input of prev layer
            # dropout=0.2,
            bias=False,
            batch_first=True
        )
        self.phead = nn.Linear(                     # This is the prediction head
            in_features=35,
            out_features=5
        )
    
    def forward(self,x):
        
        output,(h_n,c_n)=self.lstm(x)

        last_hidden = h_n[-1]
        weights = self.phead(last_hidden)

        softmax = nn.Softmax(dim=1)
        weights = softmax(weights)
        # print(weights)
        weights = weights[-1]


        return weights