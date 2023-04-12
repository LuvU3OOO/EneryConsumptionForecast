import numpy as np
import torch
import torch.nn as nn
from dataset import train_loader, X_train, week_train_dataset, week_test_dataset, y_test, test_loader, batch_size
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
import datetime

# parameters
input_dim = X_train.shape[2]
n_seq = 7
batch_size = batch_size
output_dim = 1
hidden_dim = 128
n_epochs = 100
num_layers = 1
learning_rate = 1e-3
weight_decay = 1e-6
is_bidirectional = False
dropout_prob = 0.1
save_path = r'./models/Model1.ckpt'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Transformer(nn.Module):
    # d_model : number of features
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=6, dropout=dropout_prob)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(input_dim, 1)
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
        mask = self._generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder(src, mask)
        output = self.decoder(output)
        return output


TfModel = Transformer().to(device)
optimizer = torch.optim.Adam(TfModel.parameters(),lr=0.001,weight_decay=weight_decay)
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
    lstm_values = np.ravel(inverse_transform(y_test.values.reshape(-1, 1), pred_transformer_values))
#   print(lstm_values.shape)
#   print(lstm_values[:10])
    df_lstm_values = format_predictions(lstm_values, y_test, y_test.index)
    print(df_lstm_values.head())
    subplots_time_series(df_lstm_values.index.to_list(), df_lstm_values["total_load_real_values"],
                         df_lstm_values["total_load_predicted_values"], "Transformer")
    plot_multiple_time_series(df_lstm_values.index.to_list(), df_lstm_values["total_load_real_values"],
                              df_lstm_values["total_load_predicted_values"], "Transformer")