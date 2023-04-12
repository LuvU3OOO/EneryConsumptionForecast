from Data import df
import numpy as np
import pandas as pd
import os
import torch
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

begin_train_date = datetime.strptime("2015-01-01", "%Y-%m-%d").date()
end_train_date = datetime.strptime("2017-12-29", "%Y-%m-%d").date()
end_train_date_1 = datetime.strptime("2018-12-28", "%Y-%m-%d").date()
end_test_date = datetime.strptime("2018-12-29", "%Y-%m-%d").date()

X_train = df.loc[(df.index > begin_train_date) & (df.index < end_train_date), ~df.columns.isin(["total_load_actual"])]
y_train = df.loc[(df.index > begin_train_date) & (df.index < end_train_date), "total_load_actual"]
X_test = df.loc[(df.index > end_train_date) & (df.index < end_test_date), ~df.columns.isin(["total_load_actual"])]
y_test = df.loc[(df.index > end_train_date) & (df.index < end_test_date), "total_load_actual"]
# X_test = df.loc[(df.index > end_train_date) & (df.index < end_test_date), ~df.columns.isin(["total_load_actual"])]
# y_test = df.loc[(df.index > end_train_date) & (df.index < end_test_date), "total_load_actual"]
print("Number of train data: ", X_train.shape)
print("Number of test data: ", X_test.shape)
print(X_train.head())

# 数据标准化
def preprocessing(dataframe):
    new_dataframe = dataframe.copy()
    scaler = StandardScaler()
    # applying StandardScaler
    if len(new_dataframe.shape) > 1:
        scaler.fit(new_dataframe)  # 计算mean,std
        # std
        new_values = scaler.transform(new_dataframe)
        new_dataframe.loc[:, new_dataframe.columns.to_list()] = new_values

        # remove cols with constant values 删掉80%不变值，对训练无用
        thr = int(dataframe.shape[0] * 0.8)
        rm_cols = []
        for col in new_dataframe.columns:
            values, counts = np.unique(new_dataframe[col], return_counts=True)
            if (len(values) == 1) or (np.any(counts > thr)):
                rm_cols.append(col)

        new_dataframe = new_dataframe.drop(labels=rm_cols, axis="columns")
        # new_df = df.drop(labels=rm_cols, axis="columns")
        # new_df.to_csv(r'C:\D\PytorchProgect\pytorchrtest\EnergyForecast\data\energy.csv')
    else:
        scaler.fit(new_dataframe.values.reshape(-1, 1))
        new_values = scaler.fit_transform(new_dataframe.values.reshape(-1, 1))
        new_dataframe = new_values

    return new_dataframe


new_X_train = preprocessing(X_train)
new_y_train = preprocessing(y_train)

new_X_test = preprocessing(X_test)
new_y_test = preprocessing(y_test)

print("Number of train columns", new_X_train.shape[1])
print("Number of test columns", new_X_test.shape[1])


# 删除了new_X_train中存在但不在new_X_test中的列，反之亦然。
def rm_unseen_cols(train_df, test_df):
    cols_not_in_train = [col for col in test_df.columns if col not in train_df.columns]
    cols_not_in_test = [col for col in train_df.columns if col not in test_df.columns]
    new_X_test_df = test_df.drop(labels=cols_not_in_train, axis="columns")
    new_X_train_df = train_df.drop(labels=cols_not_in_test, axis="columns")

    return new_X_train_df, new_X_test_df


new_X_train, new_X_test = rm_unseen_cols(new_X_train, new_X_test)

print("Number of train columns:", new_X_train.shape[1])
print("Number of test columns:", new_X_test.shape[1])
# print(new_X_train.columns)

# train dataframe
train_dataframe = new_X_train
# our target
train_dataframe["total_load_actual"] = new_y_train

# test dataframe
test_dataframe = new_X_test
# our target
test_dataframe["total_load_actual"] = new_y_test

"""
由于我们要预测未来7天的能源需求，我们需要从我们的数据中创建一个历史。这就是监督函数的作用。我们将通过n个输入日来分离X_train值，
y_train值将是从输入的最后一天开始的下一个n个输出日。然后，我们将创建训练数据加载器，以保存X_train和y_train信息。
同时，我们要创建一个week_test_dataset和week_train_dataset，将数据按周分批。这对以后评估模型很有用。
"""
# 为gpu生成种子，让每次实验结果一致
torch.cuda.manual_seed(100)

sequence_length = 7  # 为了预测未来7天能源消耗
batch_size = 20

features = [col for col in train_dataframe.columns if col != "total_load_actual"]
target = "total_load_actual"


def create_dataset(train_dataframe, test_dataframe, target, n_input, n_out):
    # convert history into inputs and outputs
    def to_supervised(dataframe, target, n_input=n_input, n_out=n_out):
        """
        This functions creates a history from the dataframe values and target values.
        For the X value we're going to separate the values from dataframe by n_input days.
        For the Y value we're going to get the values from the last day of input (in_end) till
        the n_out (that is the number of outputs).
        Args:
        Dataframe: pd.Dataframe. the dataframe with all the values including the values from the target
        Target: string. Name of the column that we're going to forecast
        N_input: int. The n_input days that are going to by our history. By default is 7.
        N_out: int. The size of sequence that we're going to forecast. By defeaut is 7.
        Returns:
        X,Y: np.array,np.array: The X vector has the history values from the dataset and the Y contains the history values
        that we're going to predicted.
    """
        X, y = list(), list()
        in_start = 0
        # step over the entire history one time step at a time
        for _ in range(dataframe.shape[0]):
            # define the end of the input sequence
            in_end = in_start + n_input
            out_end = in_end + n_out
            # ensure we have enough data for this instance
            if out_end <= dataframe.shape[0]:
                x_input = dataframe.iloc[in_start:in_end, :].values
                X.append(x_input)
                y.append(dataframe[target].iloc[in_end:out_end].values)
            # move along one time step
            in_start += 1
        return np.array(X), np.array(y)

    xtrain, ytrain = to_supervised(train_dataframe, target)
    xtest, ytest = to_supervised(test_dataframe, target)

    train_dataset = TensorDataset(torch.Tensor(xtrain), torch.Tensor(ytrain).unsqueeze(2))   # size(1079,7,1)-》(1079,7)
    test_dataset = TensorDataset(torch.Tensor(xtest), torch.Tensor(ytest).unsqueeze(2))
    return train_dataset, test_dataset


def split_dataset(dataframe):
    # split into standard weeks
    sp_df = np.array(np.split(dataframe.values, dataframe.shape[0] / 7))
    return sp_df


train_dataset, test_dataset = create_dataset(train_dataframe, test_dataframe, target, n_input=7, n_out=7)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
X_train, y_train = next(iter(train_loader))

print("Train features shape:", X_train.shape)
print("Train target shape:", y_train.shape)

# split the dataframe by weeks
week_train_dataset = split_dataset(train_dataframe)
week_test_dataset = split_dataset(test_dataframe)

print("Week train dataframe shape:", week_train_dataset.shape)
print("Week test dataframe  shape:", week_test_dataset.shape)

