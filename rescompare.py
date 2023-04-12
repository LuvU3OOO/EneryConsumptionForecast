import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import metrics
from RnnModel import df_rnn_values
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler
import numpy as np

loss_path = r'C:\D\PytorchProgect\pytorchrtest\EnergyForecast\data\loss'
pred_path = r'C:\D\PytorchProgect\pytorchrtest\EnergyForecast\data\pred'


def getFiles(dir_path):
    filepath = []
    for file in os.listdir(dir_path):  # 不仅仅是文件，当前目录下的文件夹也会被认为遍历到
        filepath.append(os.path.join(dir_path, file))
    return filepath


loss_list = getFiles(loss_path)
pred_list = getFiles(pred_path)


def pltLoss():
    data = pd.DataFrame()
    plt.rcParams.update({"font.size": 20})
    for f in loss_list:
        df = pd.read_csv(f)
        data = pd.concat([data, df], axis=1)
    # print(data)

    models = list(data.columns)
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    fig, ax = plt.subplots()
    for i in models:
        ax.plot(data[i].values, label=i)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title('loss compare')
    ax.legend(fontsize=12)
    plt.savefig('loss1.svg', format='svg')
    plt.show()

    fig1, ax2 = plt.subplots()
    ax2.plot(data['Transformer-Bilstm'].values, label='Transformer-Bilstm')
    ax2.plot(data['Transformer-Lstm'].values, label='Transformer-Lstm')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.set_title('Train loss')
    ax2.legend(fontsize=12)
    plt.savefig('loss2.svg', format='svg')
    plt.show()


def pltPred():
    data = pd.DataFrame()
    for f in pred_list:
        df = pd.read_csv(f)
        data = pd.concat([data, df], axis=1)
    # print(data)
    index = df_rnn_values.index.to_list()
    models = list(data.columns)
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams.update({"font.size": 20})
    fig, ax = plt.subplots(2, 1, figsize=(20, 10))
    for i in models:
        if i == 'real':
            ax[0].plot(index, data[i].values, ".-y", label='Total_Real_Load', linewidth=1)
        elif i == 'Rnn' or i == 'Gru' or i == 'Lstm':
            ax[1].plot(index, data[i].values, label=i)
        else:
            ax[0].plot(index, data[i].values, label=i)
    ax[0].set_ylabel('Load')
    ax[0].set_title('Real and Predicted 7 Days Load Forecast')
    ax[0].legend(fontsize=12)

    ax[1].plot(index, data['real'].values, ".-y", label='Total_Real_Load', linewidth=1)
    ax[1].set_ylabel('Load')
    ax[1].set_title('Real and Predicted 7 Days Load Forecast')
    ax[1].legend(fontsize=12)
    plt.xticks(rotation=45)
    plt.savefig('pred1.svg', format='svg')
    plt.show()

    fig1, ax2 = plt.subplots(figsize=(20, 10))
    ax2.plot(index, data['Transformer-Bilstm'].values, ".-r", label='Transformer-Bilstm', linewidth=1)
    ax2.plot(index, data['real'].values, ".-y", label='Total_Real_Load', linewidth=1)

    ax2.set_ylabel('Load')
    ax2.set_title('Real and Predicted 7 Days Load Forecast')
    plt.xticks(rotation=45)
    ax2.legend(fontsize=12)
    plt.savefig('pred2.svg', format='svg')
    plt.show()

    fig2, ax3 = plt.subplots(figsize=(20, 10))
    ax3.plot(index, data['Transformer-Lstm'].values, ".-r", label='Transformer-Lstm', linewidth=1)
    ax3.plot(index, data['real'].values, ".-y", label='Total_Real_Load', linewidth=1)

    ax3.set_ylabel('Load')
    ax3.set_title('Real and Predicted 7 Days Load Forecast')
    plt.xticks(rotation=45)
    ax3.legend(fontsize=12)
    plt.savefig('pred3.svg', format='svg')
    plt.show()


def norm(dataframe):
    new_dataframe = dataframe.copy()
    scaler = StandardScaler()
    scaler.fit(new_dataframe)  # 计算mean,std
    # std
    new_values = scaler.transform(new_dataframe)
    new_dataframe.loc[:, new_dataframe.columns.to_list()] = new_values
    return new_dataframe


def Mape(y_true, y_pred):
    res_mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100

    return res_mape


def evaluate():
    data = pd.DataFrame()
    for f in pred_list:
        df = pd.read_csv(f)
        data = pd.concat([data, df], axis=1)
    # print(data)
    data = norm(data)
    models = list(data.columns)
    models.remove('real')
    eva = pd.DataFrame(columns=models)
    real = data['real']
    for i in models:

        s = 0
        for j in range(len(real)):
            s += (real[j] - data[i][j]) ** 2
        rmse = sqrt(s / len(real))

        res_mae = metrics.mean_absolute_error(real, data[i])

        eva.loc['rmse', i] = np.round(rmse, 2)
        eva.loc['mae', i] = np.round(res_mae, 2)

    eva.to_csv(r'C:\D\PytorchProgect\pytorchrtest\EnergyForecast\data\eva.csv')

pltLoss()

pltPred()
# evaluate()
