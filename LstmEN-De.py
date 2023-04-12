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
n_epochs = 200
num_layers = 2
learning_rate = 1e-3
weight_decay = 1e-6
is_bidirectional = False
dropout_prob = 0.2
save_path = r'./models/E_DModel2.ckpt'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if is_bidirectional:
    D = 2
else:
    D = 1


class LSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = num_layers
        self.bidirectional = is_bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=is_bidirectional, dropout=dropout_prob
        )

    def forward(self, x):
        h0 = torch.zeros(num_layers * D, x.size(0), hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(num_layers * D, x.size(0), hidden_dim).requires_grad_().to(device)

        # we need to detach since we're doing backpropagatio through time
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        return hn, cn


class LSTMDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Defining the number of layers and the nodes in each layer
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = num_layers
        self.bidirectional = is_bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=is_bidirectional, dropout=dropout_prob
        )
        self.fc = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x, hn, cn):
        # we need to detach since we're doing backpropagatio through time
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out_lstm, (hn, cn) = self.lstm(x, (hn, cn))
        out_lstm = out_lstm[:, -1, :]
        out = self.fc(out_lstm)
        return out, hn, cn


lstm_encoder = LSTMEncoder().to(device)
lstm_decoder = LSTMDecoder().to(device)


class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.encoder = lstm_encoder
        self.decoder = lstm_decoder
        self.learning_rate = learning_rate


LstmEn_De = EncoderDecoder().to(device)
criterion = torch.nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(params=LstmEn_De.parameters(), lr=learning_rate, weight_decay=weight_decay)

train_losses = []
val_losses = []


def train():
    for epoch in range(n_epochs):
        batch_losses = []
        for i, (inputs, label) in enumerate(train_loader):
            outputs = torch.zeros(label.shape[1], label.shape[0])
            inputs = inputs.to(device)
            optimizer.zero_grad()
            encoder_hidden, encoder_cell = LstmEn_De.encoder(inputs)
            decoder_input = inputs.to(device)
            decoder_hidden = encoder_hidden.to(device)
            decoder_cell = encoder_cell.to(device)
            for t in range(inputs.shape[1]):
                decoder_output, decoder_hidden, decoder_cell = LstmEn_De.decoder(decoder_input, decoder_hidden, decoder_cell)
                outputs[t] = decoder_output.view([decoder_output.shape[1], decoder_output.shape[0]])

            # compute the loss
            label = label.view([label.shape[1], label.shape[0]])
            loss = criterion(outputs, label)
            batch_losses.append(loss.item())

            # backpropagation
            loss.backward()
            optimizer.step()

        # adding batch_loss to train_losses
        training_loss = np.mean(batch_losses)
        train_losses.append(training_loss)

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f'[{epoch + 1}/{n_epochs}] Training loss: {training_loss:.4f}')

    # 保存模型
    if not os.path.isdir('models'):
        os.mkdir('models')  # Create directory of saving models.
    torch.save(LstmEn_De.state_dict(), save_path)


def forecast(history, n_input, n_seq_out):
    history = np.array(history)
    data = history.reshape(history.shape[0] * history.shape[1], history.shape[2])
    # retrieve last observations for input data
    input_x = data[-n_input:, :]
    # reshape into [1, n_input, n_input_features]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # forecast the next week
    with torch.no_grad():
        input_data = torch.Tensor(input_x).to(device)
        encoder_hidden, encoder_cell = LstmEn_De.encoder(input_data)
        # initialize tensor for predictions
        outputs = torch.zeros(input_data.shape[1], input_data.shape[0])

        # decode input_tensor
        decoder_input = input_data
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        for t in range(n_seq_out):
            decoder_output, decoder_hidden, decoder_cell = LstmEn_De.decoder(decoder_input, decoder_hidden, decoder_cell)
            outputs[t] = decoder_output.view([decoder_output.shape[1], decoder_output.shape[0]])

        outputs = outputs.cpu()
        pred = outputs.detach().numpy()
        pred = pred.reshape((pred.shape[1], pred.shape[0]))
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


def evaluate_model(train, test, n_input, n_seq_out):
    # history is a list of weekly data
    history = [x_train for x_train in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        pred_sequence = forecast(history, n_input, n_seq_out)
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

    t0 = time.time()
    train()
    t1 = time.time()
    training_time = t1 - t0
    training_time = format_time(training_time)
    print('Lstm training time:', training_time)
    plot_losses()
    score, scores, predictions = evaluate_model(week_train_dataset, week_test_dataset, n_seq, n_seq)
    summarize_scores('LSTMEn-De', score, scores)

    lstm_values = np.ravel(inverse_transform(y_test.values.reshape(-1, 1), predictions))
    print(lstm_values.shape)
    print(lstm_values[:10])
    df_lstm_values = format_predictions(lstm_values, y_test, y_test.index)
    print(df_lstm_values.head())
    subplots_time_series(df_lstm_values.index.to_list(), df_lstm_values["total_load_real_values"],
                         df_lstm_values["total_load_predicted_values"], "LSTMEn-De")
    plot_multiple_time_series(df_lstm_values.index.to_list(), df_lstm_values["total_load_real_values"],
                              df_lstm_values["total_load_predicted_values"], "LSTMEn-De")


def record():
    df = pd.DataFrame({'E-D': train_losses})  # 创建dataframe
    df1 = pd.DataFrame({'E-D': df_lstm_values["total_load_predicted_values"]})  # 创建dataframe
    df.to_csv(r'C:\D\PytorchProgect\pytorchrtest\EnergyForecast\data\loss\E-D.csv', index=False)
    df1.to_csv(r'C:\D\PytorchProgect\pytorchrtest\EnergyForecast\data\pred\E-D.csv', index=False)


record()
