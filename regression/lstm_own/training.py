import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from lstm import AirModel
import torch.optim as optim
import torch.utils.data as data
from torch import nn
from tqdm import tqdm

df = pd.read_csv('lorry_data.csv')
timeseries = df[['lorry_free']].values.astype('float32')
plt.xlabel('Time in minutes since 10.06.2022 14:58')
plt.ylabel('Lorry free')
plt.plot(timeseries)
plt.show()

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


lookback = 1
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)

model = AirModel().cuda()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

n_epochs = 2000
for epoch in tqdm(range(n_epochs)):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch.cuda())
        loss = loss_fn(y_pred.cuda(), y_batch.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train.cuda())
        train_rmse = np.sqrt(loss_fn(y_pred.cuda(), y_train.cuda()))
        y_pred = model(X_test.cuda())
        test_rmse = np.sqrt(loss_fn(y_pred.cuda(), y_test.cuda()))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))



with torch.no_grad():
    # shift train predictions for plotting
    train_plot = np.ones_like(timeseries) * np.nan
    y_pred = model(X_train)
    y_pred = y_pred[:, -1, :]
    train_plot[lookback:train_size] = model(X_train)[:, -1, :]
    # shift test predictions for plotting
    test_plot = np.ones_like(timeseries) * np.nan
    test_plot[train_size+lookback:len(timeseries)] = model(X_test)[:, -1, :]
# plot
plt.plot(timeseries, c='b')
plt.plot(train_plot, c='r')
plt.plot(test_plot, c='g')
plt.show()
