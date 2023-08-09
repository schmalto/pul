import torch.nn as nn


class AirModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, num_layers=1, bidirectional=True, dense=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dense = dense
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional)
        self.act1 = nn.SELU()
        if not bidirectional:
            self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.final = nn.Linear(self.hidden_dim, 1)
        else:
            self.linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            self.final = nn.Linear(self.hidden_dim * 2, 1)
        if bidirectional and not dense:
            self.final = nn.Linear(self.hidden_dim * 2, 1)
        else:
            self.final = nn.Linear(self.hidden_dim, 1)
        self.act2 = nn.SELU()

    def forward(self, x):
        x = x.to(self.lstm.weight_ih_l0.dtype)  # Convert x to the same data type as LSTM weights
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.act1(x)
        if self.dense:
            x = self.linear(x)
            x = self.act2(x)
        x = self.final(x)
        return x
