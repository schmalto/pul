import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from lstm import AirModel
import torch.optim as optim
import torch.utils.data as data
from torch import nn
from tqdm import tqdm
from lstm_utils import weeks, days, hours, months

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lookback = weeks(1)
hidden_dim = 64
num_layers = 2
bidirectional = True
dense = True
input_dim = 1


df = pd.read_csv('lorry_data.csv')
timeseries = df[['lorry_free']].values.astype('float32')
# plt.xlabel('Time in minutes since 10.06.2022 14:58')
# plt.ylabel('Lorry free')
# plt.plot(timeseries)
# plt.show()

train_size = int(len(timeseries) * 0.67)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]


def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i:i + lookback]
        target = dataset[i + 1:i + lookback + 1]
        X.append(feature)
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X), torch.tensor(y)


X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)

model = AirModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, bidirectional=bidirectional, dense=dense).to(device)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

training = False

if training:
    n_epochs = 2000
    for epoch in tqdm(range(n_epochs)):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch.to(device))
            loss = loss_fn(y_pred.to(device), y_batch.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        if epoch % 100 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train.to(device))
            train_rmse = np.sqrt(loss_fn(y_pred.to(device), y_train.to(device)))
            y_pred = model(X_test.to(device))
            test_rmse = np.sqrt(loss_fn(y_pred.to(device), y_test.to(device)))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

model = torch.load('ltsm.pt')
model.to(device)
model.eval()
with torch.no_grad():
    # shift train predictions for plotting
    train_plot = np.ones_like(timeseries) * np.nan
    y_pred = model(X_train.to(device))
    train_plot[lookback:train_size] = torch.Tensor.cpu(y_pred).numpy()

    # shift test predictions for plotting
    test_plot = np.ones_like(timeseries) * np.nan
    y_pred_test = model(X_test.to(device))
    test_plot[train_size + lookback:len(timeseries)] = torch.Tensor.cpu(y_pred_test).numpy()

# get last 20 values from test set
prediction_minutes = months(1)
n = days(31)
test_values = X_test[-n:]

pred_plot = np.ones((len(timeseries) + len(test_values), 1)) * np.nan
test_values = torch.tensor(test_values).to(device)

for _ in tqdm(range(prediction_minutes)):
    with torch.no_grad():
        y_pred = model(test_values.to(device))

        try:
            y_pred = y_pred[-lookback]
        except IndexError:
            print(y_pred.shape)
            print(test_values.shape)

        y_pred = y_pred[:, None, None]
        torch.cat((test_values, y_pred), dim=0)
        test_values = test_values[0:, :, :]

np.save('predictions.npy', torch.Tensor.cpu(test_values).numpy())
pred_plot[len(timeseries):] = np.squeeze(torch.Tensor.cpu(test_values).numpy(), axis=2)

# plot
plt.plot(timeseries, c='b')
plt.plot(train_plot, c='r')
plt.plot(test_plot, c='g')
plt.plot(pred_plot, c='y')
plt.savefig('lstm.png')
