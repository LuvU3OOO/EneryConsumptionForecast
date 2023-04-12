import datetime
import math
import time

import numpy as np
import torch
import torch.nn as nn
from dataset import train_loader, X_train, week_train_dataset, week_test_dataset, y_test, batch_size
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler
import pandas as pd

# parameters
input_dim = X_train.shape[2]
n_seq = 7
batch_size = batch_size
output_dim = 1
hidden_dim = 128
n_epochs = 200
num_layers = 1
learning_rate = 1e-3
weight_decay = 1e-6
is_bidirectional = False
dropout = 0.2
d_model = 32
save_path = r'./models/T-l.ckpt'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if is_bidirectional:
    D = 2
else:
    D = 1

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):  # ninp, dropout
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # 5000 * 200
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [[0],[1],...[4999]] 5000 * 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(
            10000.0) / d_model))  # e ^([0, 2,...,198] * -ln(10000)(-9.210340371976184) / 200) [1,0.912,...,(1.0965e-04)]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # 5000 * 1 * 200, 最长5000的序列，每个词由1 * 200的矩阵代表着不同的时间
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size()[0], :]  # torch.Size([35, 1, 200])
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self):  # d_model 表示特征维度（必须是head的整数倍）, num_layers 表示 Encoder_layer 的层数， dropout 用于防止过你和
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        self.fc = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)  # 位置编码前要做归一化，否则捕获不到位置信息
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout)  # 这里用了八个头
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        # self.fc1 = nn.Linear(d_model, 32)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=1,
            dropout=dropout,
            batch_first=True,
            bidirectional=is_bidirectional
        )
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim*D, 32)
        self.decoder = nn.Linear(32, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        src = self.fc(src)
        src = self.pos_encoder(src)
        mask = self._generate_square_subsequent_mask(len(src)).to(device)

        x = self.transformer_encoder(src)
        # x = self.fc1(x)
        hidden_0 = torch.zeros(1*D, x.size(0), hidden_dim).to(device)
        c_0 = torch.zeros(1*D, x.size(0), hidden_dim).to(device)

        output, (h_n, c_n) = self.lstm(x, (hidden_0.detach(), c_0.detach()))
        output = self.relu(self.fc2(output))
        # output = self.fc2(output)
        output = self.decoder(output)
        return output






TfModel = Transformer().to(device)
optimizer = torch.optim.Adam(TfModel.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = torch.nn.MSELoss(reduction="mean")

train_losses = []
val_losses = []


def train():
    for epoch in range(n_epochs):
        batch_losses = []
        for i, (inputs, label) in enumerate(train_loader):
            inputs, label = inputs.to(device), label.to(device)
            y_pred = TfModel(inputs)

            loss = criterion(y_pred, label)
            batch_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f'[{epoch + 1}/{n_epochs}] Training loss: {train_loss:.4f}')

            # 保存模型
    if not os.path.isdir('models'):
        os.mkdir('models')  # Create directory of saving models.
    torch.save(TfModel.state_dict(), save_path)


def forecast(history, n_seq):
    history = np.array(history)
    data = history.reshape(-1, history.shape[2])

    # 检索输入数据的最后观测值
    input_x = data[-n_seq:, :]

    # reshape into [1, n_input, 1]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))

    # forecast the nest week
    with torch.no_grad():
        pred = TfModel(torch.Tensor(input_x).to(device))

    pred = pred.cpu().numpy()
    # print(pred.shape)
    # print(pred[0])
    return pred[0]


def evaluate_forecasts(actual, predicted):
    scores = list()
    if len(predicted.shape) > 2:
        predicted = predicted.squeeze(axis=2)

    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


def evaluate_model(train, test, n_seq):
    # history is a list of weekly data
    history = [x_train for x_train in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        pred_sequence = forecast(history, n_seq)
        # store the predictions
        predictions.append(pred_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
        # evaluate predictions days for each week
    predictions = np.array(predictions)
    score, scores = evaluate_forecasts(test[:, :, test.shape[2] - 1], predictions)
    return score, scores, predictions


def plot_losses():
    plt.plot(train_losses, label="Training loss")
    plt.legend()
    plt.title("Losses")
    plt.show()
    plt.close()


# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


# inverse transform to results
def inverse_transform(base_values, to_transform_values):
    scaler = StandardScaler()
    scaler.fit(base_values)
    new_values = scaler.inverse_transform(to_transform_values)
    return new_values


def format_predictions(predictions, values, idx_test):
    df_res = pd.DataFrame(data={"total_load_predicted_values": predictions,
                                "total_load_real_values": values}, index=idx_test)
    return df_res


def plot_multiple_time_series(index, real_values, predicted_values, name_model):
    plt.figure(figsize=(20, 10))
    plt.plot(index, real_values, ".-y", label="real", linewidth=2)
    plt.plot(index, predicted_values, ".-.r", label="predicted", linewidth=1)
    plt.legend()
    plt.xticks(rotation=45)
    plt.title(f"{name_model} - Real x Predicted 7 Days Load Forecast")
    plt.show()
    plt.close()


def subplots_time_series(index, real_values, predicted_values, name_model):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(20, 10))
    ax[0].plot(index, real_values, ".-y", label="real", linewidth=1)
    ax[1].plot(index, predicted_values, ".-.r", label="predicted", linewidth=1)

    ax[0].legend()
    ax[1].legend()
    plt.xticks(rotation=45)
    plt.suptitle(f"{name_model} - Real and Predicted 7 Days Load Forecast")
    plt.show()
    plt.close()


def format_time(time):
    elapsed_rounded = int(round((time)))
    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


if __name__ == '__main__':
    predictions_by_model = []
    t0 = time.time()
    train()
    t1 = time.time()
    training_time = t1 - t0
    training_time = format_time(training_time)
    print('Transformer training time:', training_time)
    plot_losses()
    score, scores, predictions = evaluate_model(week_train_dataset, week_test_dataset, n_seq)
    predictions_by_model.append(predictions)
    summarize_scores('Transformer score,scores:', score, scores)
    pred_transformer_values = predictions_by_model[0].squeeze(2)
    transformer_values = np.ravel(inverse_transform(y_test.values.reshape(-1, 1), pred_transformer_values))
    print(transformer_values.shape)
    print(transformer_values[:10])
    df_transformer_values = format_predictions(transformer_values, y_test, y_test.index)
    print(df_transformer_values.head())
    subplots_time_series(df_transformer_values.index.to_list(), df_transformer_values["total_load_real_values"],
                         df_transformer_values["total_load_predicted_values"], "Transformer")
    plot_multiple_time_series(df_transformer_values.index.to_list(), df_transformer_values["total_load_real_values"],
                              df_transformer_values["total_load_predicted_values"], "Transformer")
def record():
    df = pd.DataFrame({'t-l': train_losses})  # 创建dataframe
    df1 = pd.DataFrame({'t-l': df_transformer_values["total_load_predicted_values"]})  # 创建dataframe
    df.to_csv(r'C:\D\PytorchProgect\pytorchrtest\EnergyForecast\data\loss\t-l.csv', index=False)
    df1.to_csv(r'C:\D\PytorchProgect\pytorchrtest\EnergyForecast\data\pred\t-l.csv', index=False)


record()