import torch
from torch import nn
import numpy as np
import time


class DenseLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layers=1, bidirectional=False, dense=False):
        super(DenseLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = lstm_layers
        self.bidirectional = bidirectional
        self.dense = dense
        # define the LSTM layer
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.layers,
                            bidirectional=self.bidirectional)
        self.act1 = nn.ReLU()
        # change linear layer inputs depending on if lstm is bidrectional
        if not bidirectional:
            self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.act2 = nn.ReLU()
        # change linear layer inputs depending on if lstm is bidrectional and extra dense layer isn't added
        if bidirectional and not dense:
            self.final = nn.Linear(self.hidden_dim * 2, 1)
        else:
            self.final = nn.Linear(self.hidden_dim, 1)

    def forward(self, inputs, labels=None):
        out = inputs.unsqueeze(1)
        out, h = self.lstm(out)
        out = self.act1(out)
        if self.dense:
            out = self.linear(out)
            out = self.act2(out)
        out = self.final(out)
        return out
    
    def fit(self, optimizer, criterion, epochs, train_dataloader, test_dataloader):
        print("{:<8} {:<25} {:<25} {:<25}".format('Epoch',
                                              'Train Loss',
                                              'Test Loss',
                                              'Time (seconds)'))
        train_losses=[]
        test_losses=[]
        for epoch in range(epochs):
            self.train()
            start = time.time()
            epoch_loss = []
            # for batch in train data
            for step, batch in enumerate(train_dataloader):
                # make gradient zero to avoid accumulation
                self.zero_grad()
                inputs, labels = batch
                # get predictions
                out = self(inputs)
                # get loss
                loss = criterion(out, labels)
                epoch_loss.append(loss.float().detach().cpu().numpy().mean())
                # backpropagate
                loss.backward()
                optimizer.step()
            test_epoch_loss = []
            end = time.time()
            self.eval()
            # for batch in validation data
            for step, batch in enumerate(test_dataloader):
                inputs, labels = batch
                # get predictions
                out = self(inputs)
                # get loss
                loss = criterion(out, labels)
                test_epoch_loss.append(loss.float().detach().cpu().numpy().mean())
            print("{:<8} {:<25} {:<25} {:<25}".format(epoch + 1,
                                                    np.mean(epoch_loss),
                                                    np.mean(test_epoch_loss),
                                                    end - start))

            # Save the losses for plotting
            train_losses.append(np.mean(epoch_loss))
            test_losses.append(np.mean(test_epoch_loss))
        return train_losses, test_losses
