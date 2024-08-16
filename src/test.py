# %%
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random

from ocpdet import CUSUM

import create_sim_data

random.seed(0)
np.random.seed(0)
# %%
####################
# データ生成
####################
# パラメータ
# weekday_means = [10, 20, 30, 20]
# weekday_range_hs = [5, 5, 10, 10]
# summer_rates = [1.1, 1.2, 1.3, 1.4]
# winter_rates = [0.9, 0.8, 0.7, 0.6]
# shift_days = [2, 10, 20]
# start_dates = [
#     datetime.datetime(2020, 10, 1),
#     datetime.datetime(2021, 4, 1),
#     datetime.datetime(2021, 10, 1)
# ]
weekday_means = [20, 30, 40, 20]
weekday_range_hs = [5, 5, 10, 10]
summer_rates = [1.5, 1.5, 1.5, 1.5]
winter_rates = [0.5, 0.5, 0.5, 0.5]
shift_days = [2, 2, 2]
start_dates = [
    datetime.datetime(2020, 10, 1),
    datetime.datetime(2021, 4, 1),
    datetime.datetime(2021, 10, 1)
]

# データ生成
date_df, date_dfs = create_sim_data.create_x_dates(
    weekday_means,
    weekday_range_hs,
    summer_rates,
    winter_rates,
    shift_days,
    start_dates)

# for i in range(4):
#     plt.figure(figsize=(15, 5))
#     plt.plot(date_dfs[i]["date"], date_dfs[i]["x"], label=str(i))
#     plt.plot(date_df["date"], date_df["x"], label="dst")
#     plt.legend()

# 真の変化点
true_change_idxs = np.where(date_df["is_change_point"].values==1)[0]
# データ
xs = date_df["x"].values


# %%
def draw_trues():
    for idx in true_change_idxs:
        date = dates[idx]
        plt.axvline(date, color="r", linestyle="solid")

# %%
####################
# CUMSUM
####################
model = CUSUM(k=1., h=2., burnin=50, mu=0., sigma=1.)
model.process(xs)
my_changepoints = model.changepoints


plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
dates = date_df["date"].values
plt.plot(dates, xs) 
for pr in my_changepoints:
    if len(dates) <= pr:
        continue
    plt.axvline(dates[pr], color="b", linestyle="dashed")
draw_trues()

plt.subplot(2, 1, 2)
plt.plot(dates, model.S, c="m", label="$S$")
plt.plot(dates, model.T, c="y", label="$T$")
plt.axhline(model.h, color="r", linestyle="-", zorder=-10)
plt.scatter(dates[model.changepoints], len(model.changepoints) * [model.h], marker="v",
            label="Alarm", color="green")
plt.xlabel("Time")
plt.ylabel("$S$ and $T$ statistics")
plt.legend()
plt.show()
# %%
