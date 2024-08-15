# %%
import numpy as np
import matplotlib.pyplot as plt
import datetime

import create_sim_data


# %%
weekday_means = [10, 20, 30, 20]
weekday_range_hs = [5, 5, 10, 10]
summer_rates = [1.1, 1.2, 1.3, 1.4]
winter_rates = [0.9, 0.8, 0.7, 0.6]
shift_days = [2, 10, 20]
start_dates = [
    datetime.datetime(2020, 12, 1),
    datetime.datetime(2021, 4, 1),
    datetime.datetime(2021, 12, 1)
]

date_df, date_dfs = create_sim_data.create_x_dates(
    weekday_means,
    weekday_range_hs,
    summer_rates,
    winter_rates,
    shift_days,
    start_dates)


for i in range(4):
    plt.figure(figsize=(15, 5))
    plt.plot(date_dfs[i]["day"], date_dfs[i]["x"], label=str(i))
    plt.plot(date_df["day"], date_df["x"], label="dst")
    plt.legend()