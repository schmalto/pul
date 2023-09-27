import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from lstm import LSTM_Model
import torch.utils.data as data
from tqdm import tqdm
from lstm_utils import weeks, days, hours, months


def load_dataset(lookback):
    df = pd.read_csv('lorry_data.csv')
    # Filter out the error-prone range
    timeseries = df[['lorry_free']].values.astype('float32')
    mask = np.ones_like(timeseries, dtype=bool)
    mask[101700:242500] = False
    timeseries = timeseries[mask]
    index = np.argwhere(timeseries==0)
    timeseries = np.delete(timeseries, index)
    timeseries = timeseries[:214600]
    # plt.xlabel('Time in minutes since 10.06.2022 14:58')
    # plt.ylabel('Lorry free')
    # plt.plot(timeseries)
    # plt.show()

    train_size = int(len(timeseries) * 0.67)
    test_size = len(timeseries) - train_size
    train, test = timeseries[:train_size], timeseries[train_size:]

    X_train, y_train = create_dataset(train, lookback=lookback)
    X_test, y_test = create_dataset(test, lookback=lookback)

    dataset = [X_train, y_train, X_test, y_test]

    return dataset, train_size, test_size, timeseries


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
    # Normalize values
    X = X / 28
    y = y / 28
    return torch.tensor(X), torch.tensor(y)

def plot_pred():
    test_values = np.load('predictions.npy')
    test_values = np.take(test_values, 0, axis=1).reshape(-1, 1)
    time_s = np.arange(0, len(test_values), 1, dtype=float)
    time_s = time_s / 10080
    plt.plot(time_s,test_values, c='g')
    plt.xlabel('Zeit in Wochen seit 19.06.2023 15:19')
    plt.ylabel('Freie Parkplätze (relativ zur Größe)')
    plt.savefig('predict.png')


def train_model(n_epochs, dataset, device, input_dim, hidden_dim, num_layers, bidirectional, dense):
    model_y = LSTM_Model(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, bidirectional=bidirectional,
                     dense=dense).to(device)
    X_train, y_train, _, _ = dataset
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=input_dim)

    model_y.model_training(device, dataset, loader, n_epochs=n_epochs)


def evaluate_model(p_dataset, p_device, p_train_size, p_lookback, p_timeseries):
    X_train = p_dataset[0]
    X_test = p_dataset[2]
    model = torch.load('ltsm_best.pt')
    model.to(p_device)
    model.eval()
    with torch.no_grad():
        # shift train predictions for plotting
        train_plot = np.ones_like(p_timeseries) * np.nan
        y_pred = model(X_train.to(p_device))
        train_plot[p_lookback:p_train_size] = torch.Tensor.cpu(y_pred).numpy().squeeze()

        # shift test predictions for plotting
        test_plot = np.ones_like(p_timeseries) * np.nan
        y_pred_test = model(X_test.to(p_device))
        test_plot[train_size + lookback:len(p_timeseries)] = torch.Tensor.cpu(y_pred_test).numpy().squeeze()
        p_timeseries = p_timeseries / 28
        train_plot = train_plot * 3
        test_plot = test_plot * 3
        np.save('train_plot.npy', train_plot)
        np.save('test_plot.npy', test_plot)
        


def predict(device, prediction_minutes, dataset, timeseries, lookback):
    X_train = dataset[0]
    y_train = dataset[1]
    X_test = dataset[2]
    model = torch.load('ltsm_best.pt')
    model.to(device)
    model.eval()
    # get last 20 values from test set
    n = days(31)
    test_values = X_test[-n:]
    pred_plot = np.ones((len(timeseries) + prediction_minutes, 1)) * np.nan
    #test_values = np.take(test_values, 0, axis=1)#.reshape(-1, 1)
    for _ in tqdm(range(prediction_minutes)):
        with torch.no_grad():
            y_pred = model(test_values[-lookback:].to(device))
            try:
                y_pred = y_pred[-lookback:, :]
                y_pred = torch.transpose(y_pred, 0, 1)
            except IndexError:
                print("-------------------")
                print(y_pred.shape)
                print(test_values.shape)
                print("-------------------")

            test_values = torch.cat((test_values.cpu(), y_pred.cpu()), dim=0)
            test_values = test_values[0:, :]
    np.save('predictions.npy', torch.Tensor.cpu(test_values).numpy())


def plot_eval(p_timeseries):
    
    train_plot  = np.load('train_plot.npy')
    test_plot = np.load('test_plot.npy')
   
    time_s = np.arange(0,len(p_timeseries), 1, dtype=float)
    time_s = time_s / 10080

    plt.plot(time_s,p_timeseries, c='b', label="Zeitreihe aus Daten")
    plt.plot(time_s,train_plot, c='r', label="Testzeitreihe")
    plt.plot(time_s,test_plot, c='g', label="Validationzeitreihe")
    plt.legend()
    plt.xlabel('Zeit in Wochen seit 10.06.2022 14:58')
    plt.ylabel('Freie Parkplätze (relativ zur Größe)')
    plt.savefig('eval.png')
    
    


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lookback = days(2)    
    hidden_dim = 16
    num_layers = 1
    bidirectional = True
    dense = True
    input_dim = lookback
    n_epochs = 100000
    prediction_minutes = days(31)
    dataset, train_size, test_size, timeseries = load_dataset(lookback)
    train_model(n_epochs, dataset, device, input_dim, hidden_dim, num_layers, bidirectional, dense)
    evaluate_model(dataset, device, train_size, lookback, timeseries)
    predict(device,prediction_minutes, dataset, timeseries, lookback)
    #plot_eval(timeseries)
    #plot_pred()
