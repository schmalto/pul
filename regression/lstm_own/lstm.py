import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import numpy as np


class AirModel(nn.Module):

    def __init__(self, input_dim=1, hidden_dim=50, num_layers=1, bidirectional=True, dense=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dense = dense
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                            batch_first=True, bidirectional=self.bidirectional)
        self.act1 = nn.ReLU()
        if not bidirectional:
            self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        if bidirectional and not dense:
            self.final = nn.Linear(self.hidden_dim * 2, 1)
        else:
            self.final = nn.Linear(self.hidden_dim, 1)
        self.act2 = nn.ReLU()

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = x.to(self.lstm.weight_ih_l0.dtype)  # Convert x to the same data type as LSTM weights
        x, _ = self.lstm(x)
        # x = x[:, -1, :]
        x = self.act1(x)
        if self.dense:
            x = self.linear(x)
            x = self.act2(x)
        x = self.final(x)
        return x

    def training_2(self, device, dataset, loader, n_epochs=1000):
        test_loss = []
        train_loss = []
        X_train = dataset[0]
        y_train = dataset[1]
        X_test = dataset[2]
        y_test = dataset[3]
        for epoch in tqdm(range(n_epochs)):
            self.train()
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                y_pred = self(X_batch.to(device))
                loss = self.loss_fn(y_pred.to(device), y_batch.to(device))
                train_loss.append(loss)
                loss.backward()
                self.optimizer.step()
            # Validation
            if epoch % 100 != 0:
                continue
            self.eval()
            with torch.no_grad():
                y_pred = self(X_train.to(device))
                train_rmse = np.sqrt(self.loss_fn(y_pred.to(device), y_train.to(device)))
                y_pred = self(X_test.to(device))
                test_rmse = np.sqrt(self.loss_fn(y_pred.to(device), y_test.to(device)))
                if test_rmse < min(test_loss):
                    torch.save(self, 'ltsm_best.pt')
                test_loss.append(test_rmse)

            print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
            torch.save(self, 'ltsm_last.pt')
