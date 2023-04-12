import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r'./data/energy.csv')
print(data[:5])
df1 = data.iloc[:, :16]
cols = list(data.columns[16:])
cols = ['datetime'] + cols
df2 = data.loc[:, cols]
df1.index = pd.to_datetime(data['datetime'])
df2.index = pd.to_datetime(data['datetime'])
df_energy = df1.resample("M").sum()
df_weather = df2.resample("M").mean()
x = df_energy.index


# encols1 = list(df_energy.columns)
#
# encols = map(lambda x: x[len('generation') + 1:] if 'generation' in x else x, encols1)
# df_energy.columns = encols


def newheatmap():
    col_select = ["generation_fossil_brown_coal/lignite", "generation_fossil_gas",
                  "generation_fossil_hard_coal",
                  "generation_fossil_oil", "generation_nuclear", "generation_solar",
                  "total_load_actual", "temp",
                  "pressure", "humidity", "wind_speed", "is_rain"]
    # new_data = df = df_energy.join(df_weather, on='datetime', how='inner')
    new_df = data[col_select]

    data_coor = new_df.corr()
    plt.figure(figsize=(10, 8), facecolor='w')  # 底色white
    ax = sns.heatmap(data_coor, square=True, annot=True, fmt='.3f',
                     linewidth=1, cmap='winter', linecolor='white', cbar=True,
                     annot_kws={'size': 8, 'weight': 'normal', 'color': 'white'},
                     cbar_kws={'fraction': 0.046, 'pad': 0.03})
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置字体
    # plt.rcParams['font.sans-serif']= ['Arial Unicode MS'] # 显示中文
    plt.xticks(rotation=80)  # x轴的标签旋转45度
    plt.savefig('heatmap_sec.svg', format='svg', dpi=600)
    plt.show()


def heatmapEnergy():
    data_coor = df_energy.corr()
    plt.figure(figsize=(10, 8), facecolor='w')  # 底色white
    ax = sns.heatmap(data_coor, square=True, annot=True, fmt='.3f',
                     linewidth=1, cmap='winter', linecolor='white', cbar=True,
                     annot_kws={'size': 8, 'weight': 'normal', 'color': 'white'},
                     cbar_kws={'fraction': 0.046, 'pad': 0.03})
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置字体
    # plt.rcParams['font.sans-serif']= ['Arial Unicode MS'] # 显示中文
    plt.xticks(rotation=80)  # x轴的标签旋转45度
    plt.savefig('heat_en.svg', format='svg', dpi=600)
    plt.show()


def heatmapWheather():
    df = df_weather
    df["total_load_actual"] = df_energy["total_load_actual"].values
    data_coor = df.corr()
    plt.figure(figsize=(10, 8), facecolor='w')  # 底色white
    ax = sns.heatmap(data_coor, square=True, annot=True, fmt='.3f',
                     linewidth=1, cmap='winter', linecolor='white', cbar=True,
                     annot_kws={'size': 8, 'weight': 'normal', 'color': 'white'},
                     cbar_kws={'fraction': 0.046, 'pad': 0.03})
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置字体
    # plt.rcParams['font.sans-serif']= ['Arial Unicode MS'] # 显示中文
    plt.xticks(rotation=80)  # x轴的标签旋转45度
    plt.savefig('heat_wh.svg', format='svg', dpi=600)
    plt.show()


def total_load():
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    # 设置画布大小
    plt.figure(figsize=(8, 4))
    # 画第1个柱
    plt.bar(x, df_energy['total_load_actual'].values, color='steelblue', width=16)
    plt.yticks(fontsize=18)
    plt.ylabel('Unit:MW')
    plt.title('total_load_actual(month)')
    plt.savefig('total.svg', format='svg')
    plt.show()


def energy():
    # 正常显示画图时出现的中文和负号
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    y = list(df_energy.columns)
    y_list = [y[0:3], y[3:6], y[6:9]]
    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))

    for i in range(rows):
        for j in range(cols):
            axes[i, j].bar(x, df_energy[y_list[i][j]], color='steelblue', width=10)
            axes[i, j].set_ylabel('Unit:MW')
            axes[i, j].set_title(y_list[i][j])

    # plt.title('各种类型生成能源部分概览/（月）')
    plt.savefig('energy.svg', format='svg')
    plt.show()


def weather():
    # 正常显示画图时出现的中文和负号
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    y = list(df_weather.columns)
    y_list = [y[0:3], y[3:6], y[6:9]]
    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))

    for i in range(rows):
        for j in range(cols):
            axes[i, j].bar(x, df_weather[y_list[i][j]], color='steelblue', width=10)
            if 'temp' in y_list[i][j]:
                axes[i, j].set_ylabel('Unit:K')
            if y_list[i][j] == 'pressure':
                axes[i, j].set_ylabel('Unit:hPa')
            if 'hum' in y_list[i][j] or 'cloud' in y_list[i][j]:
                axes[i, j].set_ylabel('Unit: %')
            if 'wind' in y_list[i][j]:
                axes[i, j].set_ylabel('Unit: m/s')
            if 'rain' in y_list[i][j]:
                axes[i, j].set_ylabel('Unit: mm/h')
            axes[i, j].set_title(y_list[i][j])

    # plt.title('各种类型天气因素按月平均部分概览')
    plt.savefig('weather.svg', format='svg')
    plt.show()


def test():
    col_select = ["generation_fossil_gas", "generation_fossil_hard_coal", "generation_fossil_oil", "temp",
                  "pressure", "humidity"]
    new_df = df_energy.join(df_weather, on='datetime', how='inner')
    s_df = new_df[col_select]
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    y_list = [col_select[0:3], col_select[3:6]]
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    for i in range(rows):
        for j in range(cols):
            axes[i, j].bar(x, s_df[y_list[i][j]], color='steelblue', width=10)
            if 'fossil' in y_list[i][j]:
                axes[i, j].set_ylabel('Unit:MW')
            if 'temp' in y_list[i][j]:
                axes[i, j].set_ylabel('Unit:K')
            if y_list[i][j] == 'pressure':
                axes[i, j].set_ylabel('Unit:hPa')
            if 'hum' in y_list[i][j] or 'cloud' in y_list[i][j]:
                axes[i, j].set_ylabel('Unit: %')

            axes[i, j].set_title(y_list[i][j])

    # plt.title('各种类型生成能源部分概览/（月）')
    plt.savefig('feature.svg', format='svg')
    plt.show()


if __name__ == '__main__':
    # heatmapEnergy()
    # heatmapWheather()
    # total_load()
    # energy()
    # weather()
    newheatmap()
    test()
