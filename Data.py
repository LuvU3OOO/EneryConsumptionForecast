import numpy as np
import pandas as pd
import os
import torch
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder

# loading data
df_energy = pd.read_csv(r'energy_dataset.csv')
df_weather = pd.read_csv(r'weather_features.csv')


# print(df_weather.columns.tolist())

def create_dataframe(dataset, date_cols=[], rename_columns=False,
                     remove_cols=False, cols_to_remove=[],
                     datecol_to_group=None, break_time_cols=False, agg_func="mean",
                     change_to_cat=False, cols_to_cat=[]):
    new_dataset = dataset.copy()

    # 将非数字型数据分成多类别的列并转化成独热向量编码
    if change_to_cat:
        ohe = OneHotEncoder().fit(new_dataset.loc[:, cols_to_cat])
        for col in cols_to_cat:
            new_dataset[f"{col}"] = pd.Categorical(new_dataset[f"{col}"])  # 提取数据唯一值用于分类
        new_cat_cols = [f"is_{c}" for c in ohe.categories_[0]]
        new_dataset.loc[:, new_cat_cols] = ohe.transform(new_dataset.loc[:, cols_to_cat]).toarray()
    # 删除列
    if remove_cols:
        new_dataset = new_dataset.drop(labels=cols_to_remove, axis=1)

    # 创建新的时间列
    if len(date_cols) > 0:
        for col in date_cols:
            new_dataset[f"{col}_date"] = [datetime.fromisoformat(d).date() for d in new_dataset[f"{col}"]]

    # 分解时间为年。月。日
    if break_time_cols:
        for col in date_cols:
            new_dataset[f"{col}_year"] = [d.year for d in new_dataset[f"{col}_date"]]
            new_dataset[f"{col}_month"] = [d.month for d in new_dataset[f"{col}_date"]]
            new_dataset[f"{col}_day"] = [d.day for d in new_dataset[f"{col}_date"]]

    if rename_columns:
        new_c = {c: c.replace(" ", "_") for c in new_dataset.columns.to_list()}
        new_dataset = new_dataset.rename(new_c, axis='columns')

    # create dataset with timestamp index 按时间分组实现拆分成时间戳一样的数据，以天为单位，数值被平均了
    new_dataframe = new_dataset.groupby([f"{datecol_to_group}_date"]).agg(agg_func)

    # rename index col
    if new_dataframe.index.name != "datetime":
        new_dataframe.index = new_dataframe.index.rename("datetime")

    return new_dataframe


# 需要删除的列
rm_cols_enrg = ["generation hydro pumped storage aggregated", "forecast solar day ahead",
                "forecast wind offshore eday ahead",
                "forecast wind onshore day ahead", "total load forecast", "price day ahead", "price actual"]
rm_cols_wth = ["city_name", "weather_id", "weather_description", "weather_icon"]
df_enrg = create_dataframe(df_energy, date_cols=["time"], rename_columns=True,
                           datecol_to_group="time", remove_cols=True, cols_to_remove=rm_cols_enrg,
                           break_time_cols=False,agg_func="sum")

df_wth = create_dataframe(df_weather, date_cols=["dt_iso"], remove_cols=True, cols_to_remove=rm_cols_wth,
                          datecol_to_group="dt_iso", break_time_cols=False, change_to_cat=True,
                          cols_to_cat=["weather_main"])

print("shape of energy dataset:", df_enrg.shape)
print("shape of weather features dataset:", df_wth.shape)

print("Years for weather data")
all_years_wth = np.array([i_wth.year for i_wth in df_wth.index])
years_wth, count_values_wth = np.unique(all_years_wth, return_counts=True)
for year_wth, count_wth in zip(years_wth, count_values_wth):
    print(f"{year_wth}: {count_wth}")

print("Years for energy data")
all_years_enrg = np.array([i_enrg.year for i_enrg in df_enrg.index])
years_enrg, count_values_enrg = np.unique(all_years_enrg, return_counts=True)
for year_enrg, count_enrg in zip(years_enrg, count_values_enrg):
    print(f"{year_enrg}: {count_enrg}")

# 将能源数据与天气数据结合
df = df_enrg.join(df_wth, on='datetime', how='inner')
l_remove = ['is_clear', 'is_clouds', 'is_drizzle', 'is_dust', 'is_fog', 'is_haze', 'is_mist', 'is_smoke', 'is_thunderstorm']
df = df.drop(labels=l_remove,axis=1)