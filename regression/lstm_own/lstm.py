import torch.nn as nn


class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        x = x.to(self.lstm.weight_ih_l0.dtype)  # Convert x to the same data type as LSTM weights
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x
