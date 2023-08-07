import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from lstm import DenseLSTM

if torch.cuda.is_available():
    device = torch.device(type='cuda')
else:
    device = torch.device(type='cpu')

TRAINING = True


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

def load_data(p_window_size=100):

    window_size = p_window_size
    
    df = pd.read_csv('/home/tobias/git_ws/pul/regression/lstm/lorry_data.csv')
    time_vals = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M')
    series = df['lorry_free'][0::1]
    series.index = time_vals[0::1]


    train = series[:-int(len(series) / 10)]
    test = series[-int(# device = torch.device(type='cuda')
len(series) / 10):]
    #X_train, y_train = split_sequence(train, window_size=24)

    # train test split
    train = series[:-int(len(series) / 10)]
    train_idx = train.index
    test = series[-int(len(series) / 10):]
    test_idx = test.index
    


    scaler = StandardScaler()
    train = pd.Series(scaler.fit_transform(train.values.reshape(-1, 1))[:, 0], index=train_idx)
    test = pd.Series(scaler.transform(test.values.reshape(-1, 1))[:, 0], index=test_idx)
    

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
    return train_data, test_data, window_size, train, test, y_train, y_test, scaler

def scale(arr):
    max_scale = 28
    min_scale = 0
    max = arr.max()
    min = arr.min()
    return (max_scale - min_scale) * (arr - min) / (max - min) + min_scale



train_data, test_data, window_size , train, test, y_train, y_test, scaler= load_data()
# get data loaders
batch_size = 32
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)


train_losses = []
test_losses = []

hidden_dim = 64
epochs = 200

# vanilla LSTM
model = DenseLSTM(window_size, hidden_dim, lstm_layers=2, bidirectional=True, dense=True)
model.to(device)

# define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()

# initate training
if TRAINING:
    train_losses, test_losses = model.fit(optimizer, criterion, epochs=epochs, train_dataloader=train_dataloader,test_dataloader=test_dataloader)
    torch.save(model.state_dict(), 'dense_birectional_lstm.pth')
#torch.load('vanilla_lstm_bw.pth')

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
plt.savefig('validation.png')


# plot training loss

# Plot the training and validation losses
plt.figure()
epochs_list = range(1, epochs + 1)
plt.plot(epochs_list, train_losses, label='Train Loss')
plt.plot(epochs_list, test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')

