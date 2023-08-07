import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import time


# function for generating the lagged matrix
def split_sequence(sequence, window_size):
    X = []
    y = []
    # for all indexes
    for i in range(len(sequence)):
        end_idx = i + window_size
        # exit condition
        if end_idx > len(sequence) - 1:
            break
        # get X and Y values
        seq_x, seq_y = sequence[i:end_idx], sequence[end_idx]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


df = pd.read_csv('jena_climate_2009_2017.csv')

time_vals = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M')
series = df['lorry_free'][0::1]
series.index = time_vals[0::1]

train = series[:-int(len(series) / 10)]
test = series[-int(len(series) / 10):]
X_train, y_train = split_sequence(train, window_size=24)

# train test split
train = series[:-int(len(series) / 10)]
train_idx = train.index
test = series[-int(len(series) / 10):]
test_idx = test.index

scaler = StandardScaler()
train = pd.Series(scaler.fit_transform(train.values.reshape(-1, 1))[:, 0], index=train_idx)
test = pd.Series(scaler.transform(test.values.reshape(-1, 1))[:, 0], index=test_idx)

window_size = 100

X_train, y_train = split_sequence(train, window_size=window_size)
X_test, y_test = split_sequence(test, window_size=window_size)

# convert train and test data to tensors
X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)
X_test = torch.tensor(X_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.float)

# use torch tensor datasets
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

# get data loaders
batch_size = 32
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)


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


train_losses = []
test_losses = []


def fit(model, optimizer, criterion):
    print("{:<8} {:<25} {:<25} {:<25}".format('Epoch',
                                              'Train Loss',
                                              'Test Loss',
                                              'Time (seconds)'))
    for epoch in range(epochs):
        model.train()
        start = time.time()
        epoch_loss = []
        # for batch in train data
        for step, batch in enumerate(train_dataloader):
            # make gradient zero to avoid accumulation
            model.zero_grad()
            inputs, labels = batch
            # get predictions
            out = model(inputs)
            # get loss
            loss = criterion(out, labels)
            epoch_loss.append(loss.float().detach().cpu().numpy().mean())
            # backpropagate
            loss.backward()
            optimizer.step()
        test_epoch_loss = []
        end = time.time()
        model.eval()
        # for batch in validation data
        for step, batch in enumerate(test_dataloader):
            inputs, labels = batch
            # get predictions
            out = model(inputs)
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

    # Save the trained model
    torch.save(model.state_dict(), 'vanilla_lstm_bw.pth')


# device = torch.device(type='cuda')
device = torch.device(type='cpu')

hidden_dim = 64
epochs = 5

# vanilla LSTM
model = DenseLSTM(window_size, hidden_dim, lstm_layers=1, bidirectional=False, dense=False)
model.to(device)

# define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()

# initate training
fit(model, optimizer, criterion)

# get predictions on validation set
model.eval()
preds = []
for step, batch in enumerate(test_dataloader):
    batch = tuple(t.to(device) for t in batch)
    inputs, labels = batch
    out = model(inputs)
    preds.append(out)

preds = [x.float().detach().cpu().numpy() for x in preds]
preds = np.array([y for x in preds for y in x])

# plot data and predictions and applying inverse scaling on the data
plt.plot(pd.Series(scaler.inverse_transform(y_train.float().detach().cpu().numpy().reshape(-1, 1))[:, 0],
                   index=train[window_size:].index), label='train values')
plt.plot(pd.Series(scaler.inverse_transform(y_test.float().detach().cpu().numpy().reshape(-1, 1))[:, 0],
                   index=test[:-window_size].index), label='test values')
plt.plot(pd.Series(scaler.inverse_transform(preds.reshape(-1, 1))[:, 0], index=test[:-window_size].index),
         label='test predictions')
plt.xlabel('Date time')
plt.ylabel('Lorry free')
plt.title('Vanilla LSTM Forecasts')
plt.legend()
plt.show()

# plot training loss

# Plot the training and validation losses
plt.figure()
epochs_list = range(1, epochs + 1)
plt.plot(epochs_list, train_losses, label='Train Loss')
plt.plot(epochs_list, test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Disable grad
with torch.no_grad():
    # Set the model to evaluation mode
    model.eval()

    # Define the input date and time
    input_date_time = "08.08.2023 15:31"

    # Convert the input date and time to a datetime object
    input_data = pd.to_datetime(input_date_time, format='%d.%m.%Y %H:%M')

    # Extract day, month, year, hour, and minute from the datetime object
    day = input_data.day
    month = input_data.month
    year = input_data.year
    hour = input_data.hour
    minute = input_data.minute

    # Create the input data as a 2D tensor with numerical features
    input_data = torch.tensor([[day, month, year, hour, minute]], dtype=torch.float32)

    # Pad the input tensor with zeros to match the expected input size of 100
    padded_input = torch.zeros((1, 100, input_data.shape[-1]))
    padded_input[:, -input_data.shape[1]:, :] = input_data

    # Reshape the padded input tensor to remove the batch dimension
    reshaped_input = padded_input.squeeze(0)

    # Make predictions using the model
    predictions = model(reshaped_input)

    # Print the predictions
    print(predictions)
